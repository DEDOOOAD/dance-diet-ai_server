from __future__ import annotations

import cv2
import logging
import numpy as np
from functools import lru_cache
from typing import Any
from ai_server.config import FOOD_CLASSIFIER_MIN_CONFIDENCE, FOOD_CLASSIFIER_MODEL_PATH, FOOD_CLASSIFIER_TOP_K

ClassificationResult = tuple[str, float]
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_classifier_model() -> Any:
    if not FOOD_CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Food classification model file does not exist: {FOOD_CLASSIFIER_MODEL_PATH}")

    from ultralytics import YOLO

    return YOLO(str(FOOD_CLASSIFIER_MODEL_PATH))


def unknown_prediction() -> ClassificationResult:
    return ("unknown", 0.0)


def unknown_predictions() -> list[ClassificationResult]:
    return [unknown_prediction()]


def to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    return float(value)


def decode_image(jpg_bytes: bytes) -> np.ndarray | None:
    if not jpg_bytes:
        return None

    encoded_image = np.frombuffer(jpg_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None

    return image


def classify_food_image_array(image: np.ndarray, model: Any) -> ClassificationResult:
    return classify_food_candidates_array(image, model, top_k=1, min_confidence=0.0)[0]


def classify_food_candidates_array(
    image: np.ndarray,
    model: Any,
    top_k: int = FOOD_CLASSIFIER_TOP_K,
    min_confidence: float = FOOD_CLASSIFIER_MIN_CONFIDENCE,
) -> list[ClassificationResult]:
    results = model.predict(source=image, verbose=False)
    if not results:
        logger.warning("food classifier returned no prediction results")
        return unknown_predictions()

    result = results[0]
    probs = result.probs
    if probs is None:
        return classify_food_detection_result(result, top_k, min_confidence)

    names = result.names or {}
    candidate_indexes = [int(index) for index in probs.top5[:top_k]]
    candidate_confidences = [to_float(conf) for conf in probs.top5conf[:top_k]]
    predictions: list[ClassificationResult] = []

    for class_index, confidence in zip(candidate_indexes, candidate_confidences):
        label = str(names.get(class_index, "unknown"))
        if label == "unknown":
            logger.warning(
                "food classifier label mapping missing: class_index=%s known_label_count=%s",
                class_index,
                len(names),
            )
            continue

        if confidence < min_confidence:
            logger.info(
                "food classifier candidate skipped below threshold: label=%s confidence=%.6f threshold=%.6f",
                label,
                confidence,
                min_confidence,
            )
            continue

        predictions.append((label, confidence))

    if predictions:
        return predictions

    top_index = int(probs.top1)
    confidence = to_float(probs.top1conf)
    label = str(names.get(top_index, "unknown"))
    if label == "unknown":
        logger.warning(
            "food classifier label mapping missing: top_index=%s known_label_count=%s",
            top_index,
            len(names),
        )

    return [(label, confidence)]


def classify_food_detection_result(
    result: Any,
    top_k: int,
    min_confidence: float,
) -> list[ClassificationResult]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        logger.warning(
            "food classifier result has no class probabilities or boxes: masks=%s",
            result.masks is not None,
        )
        return unknown_predictions()

    names = result.names or {}
    class_indexes = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else list(boxes.cls)
    confidences = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else list(boxes.conf)
    detections = sorted(
        zip(class_indexes, confidences),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    predictions: list[ClassificationResult] = []
    for class_index, confidence_value in detections:
        confidence = to_float(confidence_value)
        label = str(names.get(int(class_index), "unknown"))

        if label == "unknown":
            logger.warning(
                "food detector label mapping missing: class_index=%s known_label_count=%s",
                int(class_index),
                len(names),
            )
            continue

        if confidence < min_confidence:
            logger.info(
                "food detector candidate skipped below threshold: label=%s confidence=%.6f threshold=%.6f",
                label,
                confidence,
                min_confidence,
            )
            continue

        predictions.append((label, confidence))

    if predictions:
        return predictions

    if detections:
        class_index, confidence_value = detections[0]
        return [(str(names.get(int(class_index), "unknown")), to_float(confidence_value))]

    return unknown_predictions()


def classify_food_image(jpg_bytes: bytes, model: Any) -> ClassificationResult:
    image = decode_image(jpg_bytes)
    if image is None:
        logger.warning("food classifier failed to decode segment image")
        return unknown_prediction()

    return classify_food_image_array(image, model)


def classify_food_candidates(jpg_bytes: bytes, model: Any) -> list[ClassificationResult]:
    image = decode_image(jpg_bytes)
    if image is None:
        logger.warning("food classifier failed to decode image")
        return unknown_predictions()

    return classify_food_candidates_array(image, model)
