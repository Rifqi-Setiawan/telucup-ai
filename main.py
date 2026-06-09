import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai_wrapper import AdaFaceWrapper
from database import FaceEmbedding, PhotoFace, SessionLocal

load_dotenv()


def parse_det_size():
    raw_size = os.getenv("FACE_DET_SIZE", "1024").lower().replace("x", ",")
    parts = [part.strip() for part in raw_size.split(",") if part.strip()]

    try:
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])

        size = int(parts[0])
        return size, size
    except (IndexError, ValueError):
        print(f"[AI SETUP] FACE_DET_SIZE tidak valid: {raw_size}. Menggunakan 1024x1024.")
        return 1024, 1024


def get_match_threshold() -> float:
    return float(os.getenv("FACE_MATCH_THRESHOLD", "0.40"))


def get_detection_threshold() -> float:
    return float(os.getenv("FACE_DETECTION_THRESHOLD", "0.50"))


def cuda_runtime_ready(available_providers) -> bool:
    if os.getenv("FACE_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        print("[AI SETUP] FACE_FORCE_CPU aktif. InsightFace akan memakai CPU.")
        return False

    if "CUDAExecutionProvider" not in available_providers:
        print("[AI SETUP] CUDAExecutionProvider tidak tersedia. InsightFace akan memakai CPU.")
        return False

    try:
        import torch

        if not torch.cuda.is_available():
            print("[AI SETUP] PyTorch tidak mendeteksi CUDA. InsightFace akan memakai CPU.")
            return False
    except Exception as exc:
        print(f"[AI SETUP] Tidak bisa mengecek CUDA PyTorch ({exc}). InsightFace akan memakai CPU.")
        return False

    if os.name == "nt":
        cuda_dll = "cublasLt64_12.dll"
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        if not any(os.path.exists(os.path.join(path_dir, cuda_dll)) for path_dir in path_dirs):
            print(
                f"[AI SETUP] {cuda_dll} tidak ditemukan di PATH. "
                "InsightFace akan memakai CPU."
            )
            return False

    return True


def create_face_analyzer():
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "InsightFace/ONNX Runtime belum tersedia. Install dependency dari requirements.txt."
        ) from exc

    available_providers = ort.get_available_providers()
    preferred_providers = ["CPUExecutionProvider"]
    if cuda_runtime_ready(available_providers):
        preferred_providers.insert(0, "CUDAExecutionProvider")

    providers = [provider for provider in preferred_providers if provider in available_providers]

    if not providers:
        providers = ["CPUExecutionProvider"]

    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
    det_size = parse_det_size()

    print(
        f"[AI SETUP] Loading InsightFace {model_name} with providers={providers}, "
        f"det_size={det_size}, ctx_id={ctx_id}..."
    )

    face_app = FaceAnalysis(name=model_name, providers=providers)
    face_app.prepare(ctx_id=ctx_id, det_size=det_size)

    print("[AI SETUP] InsightFace RetinaFace detector loaded successfully!")
    return face_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[AI SETUP] Starting application lifespan...")

    try:
        app.state.face_app = create_face_analyzer()
    except Exception as e:
        print(f"[AI SETUP] Warning: InsightFace detector tidak berhasil dimuat. {e}")
        app.state.face_app = None

    try:
        app.state.adaface = AdaFaceWrapper(
            weight_path="weights/adaface_ir50_webface4m.ckpt",
            architecture="ir_50",
        )
    except Exception as e:
        print(f"[AI SETUP] Warning: AdaFace weight not loaded. Make sure the path is correct. {e}")
        app.state.adaface = None

    yield

    print("[AI SETUP] Shutting down application...")
    app.state.face_app = None
    app.state.adaface = None


app = FastAPI(title="Telucup Face Recognition Engine", lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class EventPhotoRequest(BaseModel):
    event_photo_id: int
    image_url: str


class RegisterFaceRequest(BaseModel):
    player_id: int
    image_url: str


class CompareFacesRequest(BaseModel):
    image_url_1: str
    image_url_2: str


def download_rgb_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Gagal mengunduh gambar dari URL. HTTP status: {response.status_code}",
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Gagal mengunduh gambar: {str(e)}")

    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid atau rusak.")


def pil_rgb_to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def detect_faces(face_app, image: Image.Image):
    if not face_app:
        raise RuntimeError("InsightFace detector belum dimuat.")

    img_bgr = pil_rgb_to_bgr(image)
    faces = face_app.get(img_bgr)
    det_threshold = get_detection_threshold()

    return [
        face
        for face in faces
        if getattr(face, "kps", None) is not None and float(face.det_score) >= det_threshold
    ]


def bbox_to_dict(bbox, image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    return {
        "x": x1,
        "y": y1,
        "w": max(0, x2 - x1),
        "h": max(0, y2 - y1),
    }


def face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def cosine_similarity(vec1, vec2) -> float:
    arr1 = np.asarray(vec1, dtype=np.float32)
    arr2 = np.asarray(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    if norm1 <= 0 or norm2 <= 0:
        return -1.0

    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def parse_embedding(raw_embedding):
    if not raw_embedding:
        return None

    if isinstance(raw_embedding, str):
        try:
            return json.loads(raw_embedding)
        except json.JSONDecodeError:
            return None

    return raw_embedding


def find_best_match(embedding_vector, known_faces):
    matched_id = None
    sim_score = -1.0

    for face in known_faces:
        db_vec = parse_embedding(face.embedding)
        if db_vec is None:
            continue

        score = cosine_similarity(embedding_vector, db_vec)
        if score > sim_score:
            sim_score = score
            matched_id = face.player_id

    return matched_id, sim_score


def process_face_recognition(photo_id: int, image_url: str, face_app, adaface) -> dict:
    """
    Background task: detect faces with RetinaFace, align each face using 5 landmarks,
    extract AdaFace embeddings, and match against registered player embeddings.
    """
    print(f"[AI WORKER] Mengunduh foto ID {photo_id} dari {image_url}...")
    db = SessionLocal()

    try:
        if not face_app:
            raise RuntimeError("InsightFace detector belum dimuat.")

        if not adaface:
            raise RuntimeError("AdaFace model belum dimuat.")

        img = download_rgb_image(image_url)
        faces = detect_faces(face_app, img)

        db.query(PhotoFace).filter(PhotoFace.event_photo_id == photo_id).delete()

        if not faces:
            print("[AI WORKER] Tidak ada wajah yang terdeteksi di foto ini.")
            db.commit()
            return {
                "event_photo_id": photo_id,
                "faces_detected": 0,
                "faces_saved": 0,
                "matched_faces": 0,
            }

        print(f"[AI WORKER] Ditemukan {len(faces)} wajah. Memulai ekstraksi dan pencocokan...")

        known_faces = db.query(FaceEmbedding).all()
        successful_faces = 0
        matched_faces = 0
        threshold = get_match_threshold()

        for i, face in enumerate(faces):
            try:
                bbox_dict = bbox_to_dict(face.bbox, img.width, img.height)

                if bbox_dict["w"] < 10 or bbox_dict["h"] < 10:
                    print(
                        f"  [SKIP] Wajah ke-{i + 1}: bounding box terlalu kecil "
                        f"({bbox_dict['w']}x{bbox_dict['h']}), melewati."
                    )
                    continue

                embedding_vector = adaface.align_and_extract(img, face.kps)
                matched_id, sim_score = find_best_match(embedding_vector, known_faces)

                if matched_id is not None and sim_score >= threshold:
                    matched_faces += 1
                    print(
                        f"  -> Cocok dengan Player ID {matched_id} "
                        f"(Kemiripan: {sim_score * 100:.2f}%)"
                    )
                else:
                    matched_id = None
                    print(
                        f"  -> Wajah tidak dikenal "
                        f"(score={sim_score:.4f}, threshold={threshold:.4f})."
                    )

                new_face = PhotoFace(
                    event_photo_id=photo_id,
                    matched_player_id=matched_id,
                    validation_status="pending" if matched_id else "rejected",
                    similarity_score=sim_score,
                    bounding_box=bbox_dict,
                    face_encoding=embedding_vector,
                )

                db.add(new_face)
                successful_faces += 1

            except Exception as face_err:
                print(f"  [ERROR] Gagal memproses wajah ke-{i + 1}: {str(face_err)}. Melewati wajah ini.")
                continue

        db.commit()
        print(
            f"[AI WORKER] Selesai memproses foto ID {photo_id}. "
            f"Berhasil menyimpan {successful_faces}/{len(faces)} wajah."
        )
        return {
            "event_photo_id": photo_id,
            "faces_detected": len(faces),
            "faces_saved": successful_faces,
            "matched_faces": matched_faces,
        }

    except HTTPException as e:
        db.rollback()
        print(f"[AI WORKER] Error: {e.detail}")
        raise
    except Exception as e:
        db.rollback()
        print(f"[AI WORKER] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal memproses foto event: {str(e)}")
    finally:
        db.close()


@app.post("/api/process-photo")
async def receive_photo_job(
    request: EventPhotoRequest,
    fastapi_req: Request,
):
    face_app = fastapi_req.app.state.face_app
    adaface = fastapi_req.app.state.adaface

    result = process_face_recognition(
        request.event_photo_id,
        request.image_url,
        face_app,
        adaface,
    )

    return {
        "status": "success",
        "message": "Foto event selesai diproses oleh AI.",
        "data": result,
    }


@app.post("/api/register-face")
async def register_face(
    request: RegisterFaceRequest,
    fastapi_req: Request,
    db: Session = Depends(get_db),
):
    """
    Real Face Enrollment. Downloads the player's profile photo, detects the
    largest face with RetinaFace, aligns by landmarks, extracts a 512D AdaFace
    vector, and upserts it into face_embeddings.
    """
    face_app = fastapi_req.app.state.face_app
    adaface = fastapi_req.app.state.adaface

    if not face_app:
        raise HTTPException(status_code=503, detail="InsightFace detector belum dimuat. Periksa konfigurasi server AI.")

    if not adaface:
        raise HTTPException(status_code=503, detail="AdaFace model belum dimuat. Periksa konfigurasi server AI.")

    img = download_rgb_image(request.image_url)
    faces = detect_faces(face_app, img)

    if not faces:
        raise HTTPException(
            status_code=422,
            detail="Tidak ada wajah yang terdeteksi di foto profil. Silakan unggah foto yang jelas menampilkan wajah.",
        )

    selected_face = max(faces, key=face_area)
    bbox_dict = bbox_to_dict(selected_face.bbox, img.width, img.height)

    if bbox_dict["w"] < 20 or bbox_dict["h"] < 20:
        raise HTTPException(
            status_code=422,
            detail="Wajah terdeteksi terlalu kecil. Silakan unggah foto close-up yang lebih jelas.",
        )

    embedding_vector = adaface.align_and_extract(img, selected_face.kps)
    embedding_list = embedding_vector if isinstance(embedding_vector, list) else embedding_vector.tolist()

    existing = db.query(FaceEmbedding).filter(FaceEmbedding.player_id == request.player_id).first()

    if existing:
        existing.embedding = embedding_list
        action = "updated"
    else:
        new_embedding = FaceEmbedding(
            player_id=request.player_id,
            embedding=embedding_list,
        )
        db.add(new_embedding)
        action = "created"

    db.commit()

    print(
        f"[REGISTER] Face embedding {action} untuk Player ID {request.player_id}. "
        f"Wajah terdeteksi: {len(faces)}, menggunakan wajah terbesar "
        f"({bbox_dict['w']}x{bbox_dict['h']}px)."
    )

    return {
        "status": "success",
        "message": f"Face embedding berhasil di-{action} untuk Player ID {request.player_id}.",
        "data": {
            "player_id": request.player_id,
            "action": action,
            "faces_detected": len(faces),
            "selected_face_size": {"width": bbox_dict["w"], "height": bbox_dict["h"]},
        },
    }


@app.post("/api/debug/compare-faces")
async def compare_faces(
    request: CompareFacesRequest,
    fastapi_req: Request,
):
    face_app = fastapi_req.app.state.face_app
    adaface = fastapi_req.app.state.adaface

    if not face_app:
        raise HTTPException(status_code=503, detail="InsightFace detector belum dimuat. Periksa konfigurasi server AI.")

    if not adaface:
        raise HTTPException(status_code=503, detail="AdaFace model belum dimuat. Periksa konfigurasi server AI.")

    img1 = download_rgb_image(request.image_url_1)
    img2 = download_rgb_image(request.image_url_2)

    faces1 = detect_faces(face_app, img1)
    faces2 = detect_faces(face_app, img2)

    if not faces1:
        raise HTTPException(status_code=422, detail="Tidak ada wajah yang terdeteksi pada image_url_1.")

    if not faces2:
        raise HTTPException(status_code=422, detail="Tidak ada wajah yang terdeteksi pada image_url_2.")

    face1 = max(faces1, key=face_area)
    face2 = max(faces2, key=face_area)

    embedding1 = adaface.align_and_extract(img1, face1.kps)
    embedding2 = adaface.align_and_extract(img2, face2.kps)
    similarity = cosine_similarity(embedding1, embedding2)

    bbox1 = bbox_to_dict(face1.bbox, img1.width, img1.height)
    bbox2 = bbox_to_dict(face2.bbox, img2.width, img2.height)

    return {
        "similarity": similarity,
        "landmarks_detected": True,
        "threshold": get_match_threshold(),
        "data": {
            "image_1": {
                "faces_detected": len(faces1),
                "det_score": float(face1.det_score),
                "bounding_box": bbox1,
            },
            "image_2": {
                "faces_detected": len(faces2),
                "det_score": float(face2.det_score),
                "bounding_box": bbox2,
            },
        },
    }


@app.get("/")
def health_check():
    return {"status": "AI Engine is running"}
