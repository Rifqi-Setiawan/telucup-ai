import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Pastikan path import ini sesuai dengan letak file net.py Anda
from models.net import build_model 
from face_aligner import align_face_rgb

class AdaFaceWrapper:
    def __init__(self, weight_path: str, architecture: str = 'ir_50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AI SETUP] Loading AdaFace model to {self.device}...")

        # 1. Inisialisasi arsitektur dasar dari net.py
        self.model = build_model(architecture)
        
        # 2. Pastikan file bobot benar-benar ada
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"File bobot tidak ditemukan di: {weight_path}")
            
        checkpoint = torch.load(weight_path, map_location=self.device)
        
        # PyTorch Lightning biasanya membungkus state_dict di dalam key 'state_dict'
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 3. CARA OFFICIAL ADAFACE (Berdasarkan inference.py bawaan repositori)
        # Ambil hanya key yang berawalan 'model.' dan buang 6 karakter pertama ('model.')
        model_statedict = {key[6:]: val for key, val in state_dict.items() if key.startswith('model.')}
        
        # 4. Masukkan bobot yang sudah dibersihkan ke dalam model
        self.model.load_state_dict(model_statedict)
        
        # 5. Kunci model ke mode evaluasi (wajib untuk inferensi)
        self.model.eval()
        self.model.to(self.device)
        print("[AI SETUP] Model weights loaded successfully!")

        # 6. Pipeline standar AdaFace (Ukuran wajib 112x112)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_features(self, aligned_face_image: Image.Image) -> list:
        """
        Menerima gambar wajah aligned 112x112 (PIL Image) RGB, mengonversi ke BGR,
        lalu mengembalikan list 512 dimensi.
        """
        # PERBAIKAN WARNA OFFICIAL: AdaFace dilatih dengan gambar BGR (OpenCV)
        face_array = np.array(aligned_face_image.convert("RGB"))
        face_bgr_array = face_array[:, :, ::-1] 
        face_bgr_image = Image.fromarray(face_bgr_array)

        # Ubah gambar ke tensor dan jalankan ekstraksi
        tensor = self.transform(face_bgr_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features, _ = self.model(tensor)
            
        return features[0].cpu().numpy().tolist()

    def align_and_extract(self, image: Image.Image, landmarks) -> list:
        """
        Menerima full image RGB + 5 landmarks, melakukan ArcFace/AdaFace alignment,
        lalu mengembalikan embedding 512 dimensi.
        """
        aligned_face = align_face_rgb(image, landmarks, output_size=112)
        return self.extract_features(aligned_face)
