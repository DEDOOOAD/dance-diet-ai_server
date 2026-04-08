from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL1_PATH = BASE_DIR / "ai_server" / "models" / "food_segmenter.pt"


@lru_cache(maxsize=1)
def load_segment_model() -> Any:
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"Food segmentation model file does not exist: {MODEL1_PATH}")

    from ultralytics import YOLO

    return YOLO(str(MODEL1_PATH))


def decode_image(jpg_bytes: bytes) -> np.ndarray | None:
    if not jpg_bytes:
        return None

    encoded_image = np.frombuffer(jpg_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    return image


def encode_image(image: np.ndarray) -> bytes | None:
    success, jpg_buffer = cv2.imencode(".jpg", image)
    if not success:
        return None

    return jpg_buffer.tobytes()


def image_crop(result, image: np.ndarray) -> list[bytes]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    height, width = image.shape[:2]
    segmented_images: list[bytes] = []

    for box in boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        jpg_bytes = encode_image(cropped)
        if jpg_bytes is None:
            continue

        segmented_images.append(jpg_bytes)

    return segmented_images


def segment_food_image(image: np.ndarray) -> list[bytes]:
    model = load_segment_model()
    result = model.predict(source=image, verbose=False)
    if not result:
        return []

    return image_crop(result[0], image)


def run_segmentation_model(jpg_bytes: bytes) -> list[bytes]:
    image = decode_image(jpg_bytes)
    if image is None:
        return []

    return segment_food_image(image)
