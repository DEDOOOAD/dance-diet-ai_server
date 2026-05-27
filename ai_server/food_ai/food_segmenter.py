from __future__ import annotations

import cv2
import logging
import numpy as np
from functools import lru_cache
from typing import Any
from ai_server.config import FOOD_SEGMENTER_MODEL_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_segment_model() -> Any:
    if not FOOD_SEGMENTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Food segmentation model file does not exist: {FOOD_SEGMENTER_MODEL_PATH}")

    from ultralytics import YOLO

    return YOLO(str(FOOD_SEGMENTER_MODEL_PATH))


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
        logger.warning("food segmenter returned no detection boxes")
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    height, width = image.shape[:2]
    segmented_images: list[bytes] = []
    image_area = height * width

    for box_index, (box, confidence, class_id) in enumerate(
        zip(boxes, confidences, class_ids),
        start=1,
    ):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "food segment skipped invalid box: index=%s xyxy=(%s,%s,%s,%s)",
                box_index,
                x1,
                y1,
                x2,
                y2,
            )
            continue

        crop_area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        logger.info(
            "food segment box: index=%s class_id=%s confidence=%.6f xyxy=(%s,%s,%s,%s) crop_area_ratio=%.6f masks=%s",
            box_index,
            int(class_id),
            float(confidence),
            x1,
            y1,
            x2,
            y2,
            crop_area_ratio,
            result.masks is not None,
        )

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        jpg_bytes = encode_image(cropped)
        if jpg_bytes is None:
            continue

        segmented_images.append(jpg_bytes)

    return segmented_images


def segment_food_image(image: np.ndarray, model: Any) -> list[bytes]:
    result = model.predict(source=image, verbose=False)
    if not result:
        return []

    return image_crop(result[0], image)


def run_segmentation_model(jpg_bytes: bytes, model: Any) -> list[bytes]:
    image = decode_image(jpg_bytes)
    if image is None:
        return []

    return segment_food_image(image, model)
