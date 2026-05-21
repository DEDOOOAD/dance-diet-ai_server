from __future__ import annotations

import time
import cv2
import mediapipe as mp
import numpy as np
from typing import Any
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_server.config import POSE_MODEL_PATH

DEFAULT_USER_WEIGHT = 60.0
DEFAULT_USER_HEIGHT = 170.0
STILL_MOVEMENT_THRESHOLD = 1.2
REFERENCE_HEIGHT = 170.0

_POSE_STATES: dict[str, dict[str, Any]] = {}


def clear_pose_session(session_id: str) -> None:
    _POSE_STATES.pop(session_id, None)


def load_pose_detector() -> Any:
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model file does not exist: {POSE_MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _next_pose_timestamp_ms(state: dict[str, Any]) -> int:
    timestamp_ms = int(time.monotonic() * 1000)
    last_pose_timestamp_ms = state["last_pose_timestamp_ms"]
    if timestamp_ms <= last_pose_timestamp_ms:
        timestamp_ms = last_pose_timestamp_ms + 1

    state["last_pose_timestamp_ms"] = timestamp_ms
    return timestamp_ms


def _calculate_movement(
    current_landmarks: list[Any],
    previous_landmarks: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    current_array = np.array([[l.x, l.y, l.z] for l in current_landmarks])

    if previous_landmarks is None:
        return 0.0, current_array

    diff = np.linalg.norm(current_array - previous_landmarks, axis=1)
    return float(np.sum(diff)), current_array


def _calculate_met(movement_score: float) -> float:
    if movement_score < STILL_MOVEMENT_THRESHOLD:
        return 1.0

    return 3.0 + min(movement_score * 2, 7.0)


def _calculate_calories(
    met: float,
    weight: float,
    elapsed_time: float,
) -> float:
    return (met * 3.5 * weight / 200) / 60 * elapsed_time


def _normalize_height_cm(height: float) -> float:
    if height <= 3:
        return height * 100

    return height


def _apply_height_adjustment(movement_score: float, height: float) -> float:
    height_cm = _normalize_height_cm(height)
    height_ratio = height_cm / REFERENCE_HEIGHT
    return movement_score * height_ratio


def analyze_pose_frame(
    frame: np.ndarray,
    *,
    detector: Any,
    session_id: str | None = None,
    frame_index: int | None = None,
    user_weight: float | None = None,
    user_height: float | None = None,
) -> dict[str, float | int]:
    if frame is None or frame.size == 0:
        return {
            "calories_burned": 0.0,
            "movement_score": 0.0,
        }

    state_key = session_id or "__default__"
    state = _POSE_STATES.setdefault(
        state_key,
        {
            "previous_landmarks": None,
            "last_pose_timestamp_ms": 0,
            "last_timestamp_ms": None,
            "total_calories": 0.0,
            "user_weight": DEFAULT_USER_WEIGHT,
            "user_height": DEFAULT_USER_HEIGHT,
        },
    )
    if user_weight is not None:
        state["user_weight"] = user_weight
    if user_height is not None:
        state["user_height"] = user_height

    timestamp_ms = _next_pose_timestamp_ms(state)
    last_timestamp_ms = state["last_timestamp_ms"]
    elapsed_time = 0.0
    if last_timestamp_ms is not None:
        elapsed_time = max((timestamp_ms - last_timestamp_ms) / 1000, 0.0)
    state["last_timestamp_ms"] = timestamp_ms

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if not detection_result.pose_landmarks:
        return {
            "calories_burned": float(state["total_calories"]),
            "movement_score": 0.0,
        }

    current_landmarks = detection_result.pose_landmarks[0]
    movement_score, current_landmarks_array = _calculate_movement(
        current_landmarks,
        state["previous_landmarks"],
    )
    state["previous_landmarks"] = current_landmarks_array

    weight = float(state["user_weight"])
    height = float(state["user_height"])
    adjusted_movement_score = _apply_height_adjustment(movement_score, height)
    current_met = _calculate_met(adjusted_movement_score)
    calories_burned = _calculate_calories(current_met, weight, elapsed_time)
    state["total_calories"] += calories_burned

    return {
        "calories_burned": float(state["total_calories"]),
        "movement_score": float(movement_score),
    }
