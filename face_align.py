"""
Face alignment ke template standar ArcFace/AdaFace (112x112).

AdaFace (dan ArcFace) dilatih pada wajah yang sudah di-align: 5 titik landmark
(mata kiri, mata kanan, hidung, sudut mulut kiri, sudut mulut kanan) di-warp via
similarity transform ke posisi kanonik pada kanvas 112x112. Tanpa langkah ini,
embedding wajah yang sedikit miring/menoleh jadi tidak konsisten dan skor
kemiripan turun drastis -> wajah asli pun gagal dikenali.

Template referensi resmi (InsightFace / AdaFace), urutannya cocok dengan output
landmark MTCNN facenet_pytorch: [left_eye, right_eye, nose, mouth_left, mouth_right].
"""
import cv2
import numpy as np
from PIL import Image

# Posisi target 5 landmark pada gambar 112x112 (standar ArcFace).
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

OUTPUT_SIZE = (112, 112)


def align_face(pil_img: Image.Image, landmark5) -> Image.Image | None:
    """
    Meng-align wajah memakai 5 landmark ke template ArcFace 112x112.

    Args:
        pil_img: PIL Image RGB (gambar penuh, bukan crop).
        landmark5: array-like bentuk (5, 2) -> (x, y) untuk
                   [mata_kiri, mata_kanan, hidung, mulut_kiri, mulut_kanan].

    Returns:
        PIL Image 112x112 RGB yang sudah di-align, atau None bila transform gagal.
    """
    src = np.asarray(landmark5, dtype=np.float32).reshape(5, 2)

    # Similarity transform (rotasi + skala seragam + translasi) - persis seperti
    # yang dipakai pipeline ArcFace untuk meng-align wajah.
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None

    img_rgb = np.array(pil_img)
    aligned = cv2.warpAffine(
        img_rgb, M, OUTPUT_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return Image.fromarray(aligned)
