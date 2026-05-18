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
STILL_MOVEMENT_THRESHOLD = 1.2

_DETECTOR: Any | None = None
_POSE_STATES: dict[str, dict[str, Any]] = {}


def clear_pose_session(session_id: str) -> None:
    _POSE_STATES.pop(session_id, None)


def _get_detector() -> Any:
    global _DETECTOR

    if _DETECTOR is None:
        base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        _DETECTOR = vision.PoseLandmarker.create_from_options(options)

    return _DETECTOR


def _calculate_movement(
    current_landmarks: list[Any],
    previous_landmarks: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    current_array = np.array([[l.x, l.y, l.z] for l in current_landmarks])

    if previous_landmarks is None:
        return 0.0, current_array

    diff = np.linalg.norm(current_array - previous_landmarks, axis=1)
    return float(np.sum(diff)), current_array


def analyze_pose_frame(
    frame: np.ndarray,
    *,
    session_id: str | None = None,
    frame_index: int | None = None,
    user_weight: float | None = None,
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
            "last_time": time.time(),
            "total_calories": 0.0,
        },
    )

    current_time = time.time()
    elapsed_time = max(current_time - state["last_time"], 0.0)
    state["last_time"] = current_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = _get_detector().detect(mp_image)

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

    if movement_score < STILL_MOVEMENT_THRESHOLD:
        current_met = 1.0
    else:
        current_met = 3.0 + min(movement_score * 2, 7.0)

    weight = user_weight or DEFAULT_USER_WEIGHT
    calories_burned = (current_met * 3.5 * weight / 200) / 60 * elapsed_time
    state["total_calories"] += calories_burned

    return {
        "calories_burned": float(state["total_calories"]),
        "movement_score": float(movement_score),
    }
