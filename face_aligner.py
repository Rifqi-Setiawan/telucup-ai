import numpy as np
import cv2
from PIL import Image


ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image must be an RGB image with 3 channels.")

    return image_array


def align_face_rgb(image, landmarks, output_size: int = 112) -> Image.Image:
    """
    Align a face from a full RGB image using 5 facial landmarks.
    Returns a 112x112 RGB PIL image ready for AdaFace preprocessing.
    """
    image_rgb = _to_rgb_array(image)
    src = np.asarray(landmarks, dtype=np.float32)

    if src.shape != (5, 2):
        raise ValueError("Expected landmarks with shape (5, 2).")

    if output_size != 112:
        scale = output_size / 112.0
        dst = ARCFACE_TEMPLATE_112 * scale
    else:
        dst = ARCFACE_TEMPLATE_112

    transform_matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform_matrix is None:
        raise ValueError("Failed to estimate face alignment transform.")

    aligned = cv2.warpAffine(
        image_rgb,
        transform_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return Image.fromarray(aligned)
