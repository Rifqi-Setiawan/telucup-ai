import io
import os
import json
import requests
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from PIL import Image
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from facenet_pytorch import MTCNN
from database import SessionLocal, PhotoFace, FaceEmbedding
from ai_wrapper import AdaFaceWrapper, resolve_device
from face_align import align_face

load_dotenv()

# ---------------------------------------------------------------------------
# Konfigurasi (semua bisa di-override via .env)
# ---------------------------------------------------------------------------
# Confidence minimum deteksi MTCNN. Deteksi di bawah ini dibuang (false positive
# / wajah terlalu blur) supaya tidak menghasilkan embedding sampah.
DET_THRESHOLD = float(os.getenv("FACE_DET_THRESHOLD", 0.90))

# Threshold pencocokan TIGA TINGKAT (cosine similarity, embedding ter-align):
#   skor >= HIGH    -> "pending"      (yakin, siap divalidasi PIC)
#   REVIEW..HIGH    -> "needs_review" (ragu, PIC cek manual)
#   skor <  REVIEW  -> "rejected"     (tidak dikenal)
MATCH_HIGH = float(os.getenv("FACE_MATCH_HIGH", 0.40))
MATCH_REVIEW = float(os.getenv("FACE_MATCH_REVIEW", 0.25))
PROCESS_ASYNC = os.getenv("FACE_PROCESS_ASYNC", "false").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[AI SETUP] Starting application lifespan...")
    device = resolve_device()
    # keep_all=True -> deteksi semua wajah; min_face_size=40 -> abaikan wajah super kecil
    app.state.mtcnn = MTCNN(keep_all=True, min_face_size=40, device=device)
    try:
        app.state.adaface = AdaFaceWrapper(
            weight_path='weights/adaface_ir50_webface4m.ckpt',
            architecture='ir_50',
            device=device,
        )
    except Exception as e:
        print(f"[AI SETUP] Warning: AdaFace weight not loaded. Make sure the path is correct. {e}")
        app.state.adaface = None

    print(f"[AI SETUP] Config -> DEVICE={device}, DET_THRESHOLD={DET_THRESHOLD}, "
          f"MATCH_HIGH={MATCH_HIGH}, MATCH_REVIEW={MATCH_REVIEW}, PROCESS_ASYNC={PROCESS_ASYNC}", flush=True)
    yield

    print("[AI SETUP] Shutting down application...")
    app.state.mtcnn = None
    app.state.adaface = None


app = FastAPI(title="Telucup Face Recognition Engine", lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EventPhotoRequest(BaseModel):
    event_photo_id: int
    image_url: str


class RegisterFaceRequest(BaseModel):
    player_id: int
    # Dukungan dua format: satu foto (image_url) atau banyak foto (image_urls).
    image_url: Optional[str] = None
    image_urls: Optional[List[str]] = None
    # replace=True -> hapus embedding lama player ini lalu simpan yang baru (default).
    # replace=False -> tambahkan embedding baru ke yang sudah ada.
    replace: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return None
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    except Exception:
        return None


def _detect_faces(img: Image.Image, mtcnn):
    """
    Deteksi wajah + landmark, lalu saring berdasarkan confidence.

    Returns list dict: {box:[x1,y1,x2,y2], prob:float, landmark:(5,2) | None}.
    """
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    faces = []
    if boxes is None:
        return faces

    for i, box in enumerate(boxes):
        prob = float(probs[i]) if probs is not None and probs[i] is not None else 0.0
        if prob < DET_THRESHOLD:
            continue
        lm = landmarks[i] if landmarks is not None else None
        faces.append({"box": box, "prob": prob, "landmark": lm})
    return faces


def _aligned_crop(img: Image.Image, face: dict) -> Optional[Image.Image]:
    """Hasilkan crop wajah 112x112 yang sudah di-align. Fallback ke crop mentah bila landmark hilang."""
    landmark = face.get("landmark")
    if landmark is not None:
        aligned = align_face(img, landmark)
        if aligned is not None:
            return aligned

    # Fallback (jarang): crop kotak + resize bila landmark tidak tersedia.
    x1, y1, x2, y2 = [int(b) for b in face["box"]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return img.crop((x1, y1, x2, y2)).resize((112, 112))


def _bbox_dict(img: Image.Image, box) -> dict:
    x1, y1, x2, y2 = [int(b) for b in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _load_reference_embeddings(db: Session):
    """Ambil semua embedding referensi -> list of (player_id, np.ndarray unit vector)."""
    refs = []
    for face in db.query(FaceEmbedding).all():
        vec = face.embedding
        if not vec:
            continue
        if isinstance(vec, str):
            try:
                vec = json.loads(vec)
            except Exception:
                continue
        arr = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(arr)
        if n == 0:
            continue
        refs.append((face.player_id, arr / n))  # simpan sudah ter-normalisasi
    return refs


# ---------------------------------------------------------------------------
# Recognition (foto dokumentasi event)
# ---------------------------------------------------------------------------
def process_face_recognition(photo_id: int, image_url: str, mtcnn, adaface):
    """Background task: deteksi semua wajah di foto event, align, ekstrak, cocokkan 1-to-N."""
    db = SessionLocal()
    try:
        print(f"[AI WORKER] Mengunduh foto ID {photo_id} dari {image_url}...", flush=True)
        img = _download_image(image_url)
        if img is None:
            print("[AI WORKER] Gagal mengunduh gambar.", flush=True)
            return

        faces = _detect_faces(img, mtcnn)
        if not faces:
            print("[AI WORKER] Tidak ada wajah (lolos confidence) yang terdeteksi.", flush=True)
            return

        db.query(PhotoFace).filter(PhotoFace.event_photo_id == photo_id).delete()
        db.flush()

        print(f"[AI WORKER] {len(faces)} wajah lolos confidence. Mulai ekstraksi & pencocokan...", flush=True)

        references = _load_reference_embeddings(db)
        print(f"[AI WORKER] {len(references)} embedding referensi tersedia.", flush=True)
        successful = 0

        for i, face in enumerate(faces):
            try:
                aligned = _aligned_crop(img, face)
                if aligned is None:
                    print(f"  [SKIP] Wajah ke-{i+1}: crop tidak valid.", flush=True)
                    continue

                if adaface:
                    embedding = adaface.extract_features(aligned)  # sudah L2-normalized
                else:
                    print("[AI WORKER] Warning: AdaFace not loaded. Memakai vektor dummy.", flush=True)
                    embedding = np.random.rand(512).tolist()

                # 1-to-N matching: cari player dengan kemiripan tertinggi (max atas
                # semua embedding milik player tsb -> mendukung multi-foto enroll).
                target = np.asarray(embedding, dtype=np.float32)
                tnorm = np.linalg.norm(target)
                best_id, best_score = None, None
                if tnorm > 0 and references:
                    target_unit = target / tnorm
                    for pid, ref_unit in references:
                        score = float(np.dot(target_unit, ref_unit))
                        if best_score is None or score > best_score:
                            best_score, best_id = score, pid

                # Three-tier
                if best_id is not None and best_score is not None and best_score >= MATCH_HIGH:
                    matched_id, status = best_id, "pending"
                    print(f"  -> Player {best_id} (kemiripan {best_score*100:.1f}%) [PENDING]", flush=True)
                elif best_id is not None and best_score is not None and best_score >= MATCH_REVIEW:
                    matched_id, status = best_id, "needs_review"
                    print(f"  -> Player {best_id} (kemiripan {best_score*100:.1f}%) [PERLU REVIEW]", flush=True)
                else:
                    matched_id, status = None, "rejected"
                    score_text = f"{best_score*100:.1f}%" if best_score is not None else "n/a"
                    print(f"  -> Tidak dikenal (skor terbaik {score_text}) [REJECTED]", flush=True)

                db.add(PhotoFace(
                    event_photo_id=photo_id,
                    matched_player_id=matched_id,
                    validation_status=status,
                    similarity_score=best_score,
                    bounding_box=_bbox_dict(img, face["box"]),
                    face_encoding=embedding,
                ))
                successful += 1

            except Exception as face_err:
                print(f"  [ERROR] Wajah ke-{i+1} gagal: {face_err}. Dilewati.", flush=True)
                continue

        db.commit()
        print(f"[AI WORKER] Selesai foto ID {photo_id}. Tersimpan {successful}/{len(faces)} wajah.", flush=True)

    except Exception as e:
        db.rollback()
        print(f"[AI WORKER] Error: {e}", flush=True)
    finally:
        db.close()


@app.post("/api/process-photo")
async def receive_photo_job(
    request: EventPhotoRequest,
    background_tasks: BackgroundTasks,
    fastapi_req: Request,
):
    mtcnn = fastapi_req.app.state.mtcnn
    adaface = fastapi_req.app.state.adaface
    if PROCESS_ASYNC:
        background_tasks.add_task(process_face_recognition, request.event_photo_id, request.image_url, mtcnn, adaface)
        return {"status": "success", "message": "Job diterima. AI sedang mengekstrak wajah di latar belakang."}

    process_face_recognition(request.event_photo_id, request.image_url, mtcnn, adaface)
    return {"status": "success", "message": "Foto selesai diproses oleh AI."}


# ---------------------------------------------------------------------------
# Enrollment (foto referensi player) - mendukung multi-foto
# ---------------------------------------------------------------------------
@app.post("/api/register-face")
async def register_face(request: RegisterFaceRequest, fastapi_req: Request, db: Session = Depends(get_db)):
    """
    Enroll wajah referensi player. Menerima satu foto (image_url) atau banyak
    foto multi-angle (image_urls). Tiap foto: deteksi wajah terbesar -> align ->
    ekstrak 512D -> simpan satu baris di face_embeddings.

    Lebih banyak foto (depan, agak menoleh) = pencocokan lebih akurat karena saat
    matching diambil kemiripan tertinggi atas semua embedding milik player.
    """
    mtcnn = fastapi_req.app.state.mtcnn
    adaface = fastapi_req.app.state.adaface

    if not adaface:
        raise HTTPException(status_code=503, detail="AdaFace model belum dimuat. Periksa konfigurasi server AI.")

    urls = list(request.image_urls) if request.image_urls else []
    if request.image_url:
        urls.append(request.image_url)
    if not urls:
        raise HTTPException(status_code=422, detail="Tidak ada foto yang dikirim (image_url / image_urls kosong).")

    # Mode replace: bersihkan embedding lama player ini lebih dulu.
    if request.replace:
        db.query(FaceEmbedding).filter(FaceEmbedding.player_id == request.player_id).delete()
        db.flush()

    saved = 0
    failed_urls = []

    for url in urls:
        img = _download_image(url)
        if img is None:
            failed_urls.append(url)
            continue

        faces = _detect_faces(img, mtcnn)
        if not faces:
            failed_urls.append(url)
            continue

        # Foto profil/enroll = orang utama -> ambil wajah terbesar.
        largest = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        aligned = _aligned_crop(img, largest)
        if aligned is None:
            failed_urls.append(url)
            continue

        embedding = adaface.extract_features(aligned)
        db.add(FaceEmbedding(player_id=request.player_id, embedding=embedding))
        saved += 1

    if saved == 0:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail="Tidak ada wajah yang terdeteksi di foto yang diunggah. Gunakan foto yang jelas menampilkan wajah.",
        )

    db.commit()
    print(f"[REGISTER] Player {request.player_id}: {saved} embedding tersimpan "
          f"(replace={request.replace}, gagal={len(failed_urls)}).")

    return {
        "status": "success",
        "message": f"{saved} foto wajah berhasil di-enroll untuk Player ID {request.player_id}.",
        "data": {
            "player_id": request.player_id,
            "embeddings_saved": saved,
            "failed_photos": len(failed_urls),
        },
    }


@app.get("/")
def health_check():
    return {"status": "AI Engine is running"}
