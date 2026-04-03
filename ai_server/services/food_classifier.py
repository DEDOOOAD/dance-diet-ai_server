from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL2_PATH = BASE_DIR / "ai_server" / "models" / "food_classifier.pt"  # 음식 판별 모델

ClassificationResult = tuple[str, float]


@lru_cache(maxsize=1)
def load_classifier_model() -> YOLO:
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"음식 분류 모델 파일이 없습니다: {MODEL2_PATH}")

    return YOLO(str(MODEL2_PATH))

def unknown_prediction() -> ClassificationResult:
    return ("unknown", 0.0)

def classify_food_image_path(image_path: str) -> ClassificationResult:
    model = load_classifier_model()
    results = model.predict(source=image_path, verbose=False)
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

def write_temp_image(jpg_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(jpg_bytes)
        return tmp.name

def remove_temp_image(image_path: str | None) -> None:
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

def classify_food_image(jpg_bytes: bytes) -> ClassificationResult:
    temp_path = write_temp_image(jpg_bytes)

    try:
        return classify_food_image_path(temp_path)
    finally:
        remove_temp_image(temp_path)
