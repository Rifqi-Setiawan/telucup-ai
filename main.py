from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import numpy as np

from database import SessionLocal, PhotoFace

app = FastAPI(title="Telucup Face Recognition Engine")

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

# --- CORE AI FUNCTION (MOCKUP UNTUK SEMENTARA) ---
def process_face_recognition(photo_id: int, image_url: str, db: Session):
    print(f"[AI WORKER] Mulai memproses foto ID {photo_id} dari URL: {image_url}")
    
    try:
        # 1. Download gambar dari Cloudinary
        response = requests.get(image_url)
        if response.status_code != 200:
            print("[AI WORKER] Gagal mengunduh gambar.")
            return

        image_bytes = response.content
        
        # ---------------------------------------------------------
        # DI SINI NANTI KITA AKAN MASUKKAN LOGIKA ADAFACE & OPENCV
        # Untuk menguji pipeline (Laravel -> FastAPI -> DB), 
        # kita buat data dummy seolah-olah AI menemukan 1 wajah.
        # ---------------------------------------------------------
        
        print("[AI WORKER] Mengekstrak wajah menggunakan AdaFace (Simulasi)...")
        
        # Simulasi output vektor 512 dimensi dari AdaFace
        dummy_vector = np.random.rand(512).tolist() 
        dummy_bbox = {"x": 100, "y": 150, "w": 200, "h": 200}
        
        # 2. Simpan hasil temuan AI ke database PostgreSQL (tabel photo_faces)
        new_face = PhotoFace(
            event_photo_id=photo_id,
            validation_status="pending", # Status awal wajib pending agar divalidasi peserta
            bounding_box=dummy_bbox,
            face_encoding=dummy_vector
        )
        
        db.add(new_face)
        db.commit()
        
        print(f"[AI WORKER] Sukses! Menyimpan 1 wajah terdeteksi ke database untuk foto ID {photo_id}.")
        
    except Exception as e:
        print(f"[AI WORKER] Error saat memproses: {str(e)}")


# --- API ENDPOINTS ---
@app.post("/api/process-photo")
async def receive_photo_job(request: EventPhotoRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Endpoint ini menerima request HTTP dari Laravel Job.
    Proses AI yang berat dimasukkan ke BackgroundTasks agar response API instan.
    """
    
    # Masukkan proses ke background worker FastAPI
    background_tasks.add_task(process_face_recognition, request.event_photo_id, request.image_url, db)
    
    return {
        "status": "success",
        "message": "Job diterima. AI sedang mengekstrak wajah di latar belakang."
    }

@app.get("/")
def health_check():
    return {"status": "AI Engine is running"}