import io
import requests
import numpy as np
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from PIL import Image
from contextlib import asynccontextmanager

from facenet_pytorch import MTCNN
from database import SessionLocal, PhotoFace, FaceEmbedding
from ai_wrapper import AdaFaceWrapper

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[AI SETUP] Starting application lifespan...")
    # Initialize MTCNN
    app.state.mtcnn = MTCNN(keep_all=True, min_face_size=40)
    # Initialize AdaFace (assuming weights are in 'weights/adaface.ckpt' or similar)
    try:
        app.state.adaface = AdaFaceWrapper(weight_path='weights/adaface.ckpt', architecture='ir_50')
    except Exception as e:
        print(f"[AI SETUP] Warning: AdaFace weight not loaded. Make sure the path is correct. {e}")
        app.state.adaface = None
        
    yield
    
    print("[AI SETUP] Shutting down application...")
    app.state.mtcnn = None
    app.state.adaface = None

app = FastAPI(title="Telucup Face Recognition Engine", lifespan=lifespan)

# Dependency untuk mendapatkan koneksi database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Schema validasi data yang dikirim oleh Laravel
class EventPhotoRequest(BaseModel):
    event_photo_id: int
    image_url: str

# Schema untuk endpoint register-face (Phase 1: Real Face Enrollment)
class RegisterFaceRequest(BaseModel):
    player_id: int
    image_url: str


def process_face_recognition(photo_id: int, image_url: str, db: Session, mtcnn, adaface):
    """
    Background task: mendeteksi semua wajah di foto event, 
    mengekstrak vektor, dan mencocokkan dengan face_embeddings.
    Phase 3: Hardened dengan per-face try/except agar satu wajah gagal tidak menghentikan loop.
    """
    print(f"[AI WORKER] Mengunduh foto ID {photo_id} dari {image_url}...")
    
    try:
        response = requests.get(image_url, timeout=15)
        if response.status_code != 200:
            print("[AI WORKER] Gagal mengunduh gambar.")
            return

        # Konversi byte gambar ke format PIL Image
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # 1. Deteksi semua wajah di dalam foto keramaian
        boxes, _ = mtcnn.detect(img)
        
        if boxes is None:
            print("[AI WORKER] Tidak ada wajah yang terdeteksi di foto ini.")
            return
            
        print(f"[AI WORKER] Ditemukan {len(boxes)} wajah. Memulai ekstraksi dan pencocokan...")

        successful_faces = 0

        for i, box in enumerate(boxes):
            try:
                # Format bounding box untuk disimpan [x, y, width, height]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Ensure coordinates are within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.width, x2)
                y2 = min(img.height, y2)
                
                # Skip invalid crops (too small or inverted coordinates)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    print(f"  [SKIP] Wajah ke-{i+1}: bounding box terlalu kecil ({x2-x1}x{y2-y1}), melewati.")
                    continue
                
                bbox_dict = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                
                # Potong gambar wajah
                face_crop = img.crop((x1, y1, x2, y2))
                
                # Ekstrak fitur dengan AdaFace
                if adaface:
                    embedding_vector = adaface.extract_features(face_crop)
                else:
                    print("[AI WORKER] Warning: AdaFace not loaded. Menggunakan vektor dummy.")
                    embedding_vector = np.random.rand(512).tolist() 
                
                # --- 2. 1-TO-N MATCHING DENGAN PGVECTOR ---
                # Ubah list Python ke format string vektor PostgreSQL: "[0.1, 0.2, ...]"
                vector_str = str(embedding_vector)
                
                sql_query = text("""
                    SELECT 
                        player_id, 
                        (1 - (embedding <=> :vector)) AS similarity_score
                    FROM face_embeddings
                    ORDER BY embedding <=> :vector ASC
                    LIMIT 1
                """)
                
                match_result = db.execute(sql_query, {"vector": vector_str}).fetchone()
                
                matched_id = None
                sim_score = None
                
                # Tentukan Threshold Similarity
                THRESHOLD = 0.60 
                
                if match_result and match_result.similarity_score is not None and match_result.similarity_score > THRESHOLD:
                    matched_id = match_result.player_id
                    sim_score = match_result.similarity_score
                    print(f"  -> Cocok dengan Player ID {matched_id} (Kemiripan: {sim_score*100:.2f}%)")
                else:
                    print("  -> Wajah tidak dikenal (Tidak lolos threshold).")

                # 3. Simpan hasil ke tabel photo_faces
                new_face = PhotoFace(
                    event_photo_id=photo_id,
                    matched_player_id=matched_id,
                    validation_status="pending" if matched_id else "rejected",
                    similarity_score=sim_score,
                    bounding_box=bbox_dict,
                    face_encoding=embedding_vector
                )
                
                db.add(new_face)
                successful_faces += 1

            except Exception as face_err:
                # Phase 3: Gracefully skip faces that fail extraction without crashing the loop
                print(f"  [ERROR] Gagal memproses wajah ke-{i+1}: {str(face_err)}. Melewati wajah ini.")
                continue
        
        db.commit()
        print(f"[AI WORKER] Selesai memproses foto ID {photo_id}. Berhasil menyimpan {successful_faces}/{len(boxes)} wajah.")
        
    except Exception as e:
        print(f"[AI WORKER] Error: {str(e)}")


@app.post("/api/process-photo")
async def receive_photo_job(
    request: EventPhotoRequest, 
    background_tasks: BackgroundTasks, 
    fastapi_req: Request, 
    db: Session = Depends(get_db)
):
    mtcnn = fastapi_req.app.state.mtcnn
    adaface = fastapi_req.app.state.adaface
    
    background_tasks.add_task(
        process_face_recognition, 
        request.event_photo_id, 
        request.image_url, 
        db, 
        mtcnn, 
        adaface
    )
    
    return {
        "status": "success",
        "message": "Job diterima. AI sedang mengekstrak wajah di latar belakang."
    }


@app.post("/api/register-face")
async def register_face(
    request: RegisterFaceRequest,
    fastapi_req: Request,
    db: Session = Depends(get_db)
):
    """
    Phase 1: Real Face Enrollment (Ground Truth Registration).
    Downloads the player's profile photo, detects the largest face,
    extracts a 512D AdaFace vector, and upserts into face_embeddings.
    This is a synchronous endpoint so Laravel gets an immediate error
    if no face is detected.
    """
    mtcnn = fastapi_req.app.state.mtcnn
    adaface = fastapi_req.app.state.adaface

    if not adaface:
        raise HTTPException(status_code=503, detail="AdaFace model belum dimuat. Periksa konfigurasi server AI.")

    # 1. Download gambar dari URL (Cloudinary)
    try:
        response = requests.get(request.image_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Gagal mengunduh gambar dari URL. HTTP status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Gagal mengunduh gambar: {str(e)}")

    # 2. Konversi ke PIL Image
    try:
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid atau rusak.")

    # 3. Deteksi wajah dengan MTCNN
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        raise HTTPException(
            status_code=422,
            detail="Tidak ada wajah yang terdeteksi di foto profil. Silakan unggah foto yang jelas menampilkan wajah."
        )

    # 4. Pilih bounding box terbesar (foto profil = orang utama, bukan background)
    largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = [int(b) for b in largest_box]

    # Pastikan koordinat dalam batas gambar
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.width, x2)
    y2 = min(img.height, y2)

    if x2 - x1 < 20 or y2 - y1 < 20:
        raise HTTPException(
            status_code=422,
            detail="Wajah terdeteksi terlalu kecil. Silakan unggah foto close-up yang lebih jelas."
        )

    # 5. Potong wajah dan ekstrak vektor 512D via AdaFace
    face_crop = img.crop((x1, y1, x2, y2))
    embedding_vector = adaface.extract_features(face_crop)

    # 6. Upsert: Cek apakah embedding sudah ada untuk player_id ini
    vector_str = str(embedding_vector)
    
    existing = db.query(FaceEmbedding).filter(
        FaceEmbedding.player_id == request.player_id
    ).first()

    if existing:
        # UPDATE embedding yang sudah ada
        db.execute(
            text("UPDATE face_embeddings SET embedding = :vec, updated_at = NOW() WHERE player_id = :pid"),
            {"vec": vector_str, "pid": request.player_id}
        )
        action = "updated"
    else:
        # INSERT embedding baru
        db.execute(
            text("INSERT INTO face_embeddings (player_id, embedding, created_at, updated_at) VALUES (:pid, :vec, NOW(), NOW())"),
            {"pid": request.player_id, "vec": vector_str}
        )
        action = "created"
    
    db.commit()

    print(f"[REGISTER] Face embedding {action} untuk Player ID {request.player_id}. "
          f"Wajah terdeteksi: {len(boxes)}, menggunakan wajah terbesar ({x2-x1}x{y2-y1}px).")

    return {
        "status": "success",
        "message": f"Face embedding berhasil di-{action} untuk Player ID {request.player_id}.",
        "data": {
            "player_id": request.player_id,
            "action": action,
            "faces_detected": len(boxes),
            "selected_face_size": {"width": x2 - x1, "height": y2 - y1}
        }
    }


@app.get("/")
def health_check():
    return {"status": "AI Engine is running"}