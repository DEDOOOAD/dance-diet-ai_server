from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL2_PATH = BASE_DIR / "ai_server" / "models" / "food_classifier.pt"

ClassificationResult = tuple[str, float]


@lru_cache(maxsize=1)
def load_classifier_model() -> Any:
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"Food classification model file does not exist: {MODEL2_PATH}")

    from ultralytics import YOLO

    return YOLO(str(MODEL2_PATH))


def unknown_prediction() -> ClassificationResult:
    return ("unknown", 0.0)


def decode_image(jpg_bytes: bytes) -> np.ndarray | None:
    if not jpg_bytes:
        return None

    encoded_image = np.frombuffer(jpg_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    return image


def classify_food_image_array(image: np.ndarray) -> ClassificationResult:
    model = load_classifier_model()
    results = model.predict(source=image, verbose=False)
    if not results:
        return unknown_prediction()

    result = results[0]
    probs = result.probs
    if probs is None:
        return unknown_prediction()

    top_index = int(probs.top1)
    confidence = float(probs.top1conf.item())
    names = result.names or {}
    return (str(names.get(top_index, "unknown")), confidence)


def classify_food_image(jpg_bytes: bytes) -> ClassificationResult:
    image = decode_image(jpg_bytes)
    if image is None:
        return unknown_prediction()

    return classify_food_image_array(image)
