from __future__ import annotations

import time
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Any
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ai_server.config import POSE_MODEL_PATH

DEFAULT_USER_WEIGHT = 60.0
DEFAULT_USER_HEIGHT = 170.0
MIN_USER_WEIGHT_KG = 20.0
MAX_USER_WEIGHT_KG = 250.0
STILL_MOVEMENT_THRESHOLD = 0.03
MAX_MOVEMENT_SCORE_FOR_DANCE_MET = 0.25
REFERENCE_HEIGHT = 170.0
RESTING_MET = 1.0
MAX_FRAME_ELAPSED_SECONDS = 1.0

_POSE_STATES: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class DanceMetLevel:
    activity_code: str
    description: str
    met: float


# MET anchors are selected from the 2024 Adult Compendium of Physical
# Activities, Dancing category. The calorie equation follows the standard
# Compendium/ACSM form: kcal/min = MET * 3.5 * body_weight_kg / 200.
# 2011 reference: Ainsworth et al., Med Sci Sports Exerc. 2011;43(8):1575-1581.
# DOI: 10.1249/MSS.0b013e31821ece12
DANCE_MET_LEVELS: tuple[DanceMetLevel, ...] = (
    DanceMetLevel("03040", "Ballroom dancing, slow", 3.0),
    DanceMetLevel("03070", "Contemporary dancing, general", 3.8),
    DanceMetLevel("03025", "Ethnic or cultural dancing", 4.5),
    DanceMetLevel("03010", "Ballet, modern, or jazz, rehearsal or class", 5.0),
    DanceMetLevel("03042", "Ballroom dance, recreational", 6.0),
    DanceMetLevel("03091", "Salsa dancing, to a video", 6.3),
    DanceMetLevel("03012", "Ballet, modern, or jazz, vigorous effort", 6.8),
    DanceMetLevel("03029", "Chinese square dance, aerobic dance", 7.3),
    DanceMetLevel("03075", "Flamenco dance", 8.5),
    DanceMetLevel("03031", "Nightclub or folk dancing, vigorous effort", 9.8),
)


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
    return float(np.mean(diff)), current_array


def _select_dance_met_level(movement_score: float) -> DanceMetLevel | None:
    if movement_score < STILL_MOVEMENT_THRESHOLD:
        return None

    intensity_ratio = min(
        max(
            (movement_score - STILL_MOVEMENT_THRESHOLD)
            / (MAX_MOVEMENT_SCORE_FOR_DANCE_MET - STILL_MOVEMENT_THRESHOLD),
            0.0,
        ),
        1.0,
    )
    level_index = round(intensity_ratio * (len(DANCE_MET_LEVELS) - 1))
    return DANCE_MET_LEVELS[level_index]


def _calculate_met(movement_score: float) -> float:
    met_level = _select_dance_met_level(movement_score)
    if met_level is None:
        return RESTING_MET

    return met_level.met


def _calculate_calories(
    met: float,
    weight: float,
    elapsed_time: float,
) -> float:
    return (met * 3.5 * weight / 200) / 60 * elapsed_time


def _normalize_weight_kg(weight: float | None) -> float:
    if weight is None:
        return DEFAULT_USER_WEIGHT

    normalized_weight = float(weight)
    if normalized_weight >= 1000:
        normalized_weight /= 1000

    if normalized_weight < MIN_USER_WEIGHT_KG or normalized_weight > MAX_USER_WEIGHT_KG:
        return DEFAULT_USER_WEIGHT

    return normalized_weight


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
            "total_calories": 0.0,
            "movement_score": 0.0,
            "elapsed_time_seconds": 0.0,
            "current_met": RESTING_MET,
            "active_met": 0.0,
            "gross_calories_burned": 0.0,
            "active_calories_burned": 0.0,
            "met_activity_code": "",
            "met_activity_description": "Resting or no movement",
            "user_weight_kg": DEFAULT_USER_WEIGHT,
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
        state["user_weight"] = _normalize_weight_kg(user_weight)
    if user_height is not None:
        state["user_height"] = user_height

    timestamp_ms = _next_pose_timestamp_ms(state)
    last_timestamp_ms = state["last_timestamp_ms"]
    elapsed_time = 0.0
    if last_timestamp_ms is not None:
        elapsed_time = min(
            max((timestamp_ms - last_timestamp_ms) / 1000, 0.0),
            MAX_FRAME_ELAPSED_SECONDS,
        )
    state["last_timestamp_ms"] = timestamp_ms

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if not detection_result.pose_landmarks:
        return {
            "calories_burned": 0.0,
            "total_calories": float(state["total_calories"]),
            "movement_score": 0.0,
            "elapsed_time_seconds": elapsed_time,
            "current_met": RESTING_MET,
            "active_met": 0.0,
            "gross_calories_burned": 0.0,
            "active_calories_burned": 0.0,
            "met_activity_code": "",
            "met_activity_description": "Pose not detected",
            "user_weight_kg": float(state["user_weight"]),
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
    met_level = _select_dance_met_level(adjusted_movement_score)
    current_met = met_level.met if met_level is not None else RESTING_MET
    active_met = max(current_met - RESTING_MET, 0.0)
    gross_calories_burned = _calculate_calories(current_met, weight, elapsed_time)
    active_calories_burned = _calculate_calories(active_met, weight, elapsed_time)
    state["total_calories"] += active_calories_burned

    return {
        "calories_burned": float(active_calories_burned),
        "total_calories": float(state["total_calories"]),
        "movement_score": float(movement_score),
        "elapsed_time_seconds": float(elapsed_time),
        "current_met": float(current_met),
        "active_met": float(active_met),
        "gross_calories_burned": float(gross_calories_burned),
        "active_calories_burned": float(active_calories_burned),
        "met_activity_code": met_level.activity_code if met_level is not None else "",
        "met_activity_description": (
            met_level.description if met_level is not None else "Resting or low movement"
        ),
        "user_weight_kg": float(weight),
    }
