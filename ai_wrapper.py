import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np # Tambahkan import ini
from models.net import build_model 

class AdaFaceWrapper:
    def __init__(self, weight_path: str, architecture: str = 'ir_50'):
        # Arsitektur 'ir_50' ini merujuk pada R50 di tabel repository
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AI SETUP] Loading AdaFace model to {self.device}...")

        self.model = build_model(architecture)
        
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_features(self, face_image: Image.Image) -> list:
        """
        Menerima gambar potongan wajah (PIL Image) RGB, mengonversi ke BGR,
        lalu mengembalikan list 512 dimensi.
        """
        # --- PERBAIKAN WARNA (RGB ke BGR) ---
        # 1. Ubah PIL Image (RGB) menjadi array angka NumPy
        face_array = np.array(face_image)
        
        # 2. Balik urutan channel di dimensi ke-3 (RGB menjadi BGR) menggunakan slicing numpy
        face_bgr_array = face_array[:, :, ::-1] 
        
        # 3. Kembalikan lagi ke format PIL Image agar bisa masuk ke transform PyTorch
        face_bgr_image = Image.fromarray(face_bgr_array)
        # ------------------------------------

        # Masukkan gambar yang sudah BGR ke pipeline
        tensor = self.transform(face_bgr_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features, _ = self.model(tensor)
            
        return features[0].cpu().numpy().tolist()