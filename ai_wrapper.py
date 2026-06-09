import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Pastikan path import ini sesuai dengan letak file net.py Anda
from models.net import build_model 


def resolve_device() -> torch.device:
    """Resolve inference device. Prefer CUDA when available, fallback to CPU."""
    requested = os.getenv("FACE_DEVICE", "auto").strip().lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")

    if requested == "cuda":
        print("[AI SETUP] FACE_DEVICE=cuda requested, but CUDA is not available. Falling back to CPU.")

    return torch.device("cpu")


class AdaFaceWrapper:
    def __init__(self, weight_path: str, architecture: str = 'ir_50', device: torch.device | None = None):
        self.device = device or resolve_device()
        print(f"[AI SETUP] Loading AdaFace model to {self.device}...")
        if self.device.type == "cuda":
            print(f"[AI SETUP] CUDA device: {torch.cuda.get_device_name(0)}")

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

    def _embed_single(self, face_image: Image.Image) -> np.ndarray:
        """Jalankan satu forward pass AdaFace pada PIL Image RGB -> vektor 512D (numpy)."""
        # PERBAIKAN WARNA OFFICIAL: AdaFace dilatih dengan gambar BGR (OpenCV)
        face_array = np.array(face_image)
        face_bgr_array = face_array[:, :, ::-1].copy()
        face_bgr_image = Image.fromarray(face_bgr_array)

        tensor = self.transform(face_bgr_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features, _ = self.model(tensor)
        return features[0].cpu().numpy()

    def extract_features(self, face_image: Image.Image) -> list:
        """
        Menerima wajah yang IDEALNYA sudah di-align (PIL Image RGB 112x112),
        lalu mengembalikan embedding 512 dimensi yang sudah di-L2-normalize.

        Memakai flip-test augmentation (teknik standar inferensi AdaFace/ArcFace):
        embedding gambar + embedding cerminannya dijumlahkan lalu dinormalisasi.
        Ini membuat embedding lebih stabil terhadap variasi pose ringan tanpa
        menambah dimensi (tetap 512).
        """
        f1 = self._embed_single(face_image)
        f2 = self._embed_single(face_image.transpose(Image.FLIP_LEFT_RIGHT))

        fused = f1 + f2
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        return fused.tolist()
