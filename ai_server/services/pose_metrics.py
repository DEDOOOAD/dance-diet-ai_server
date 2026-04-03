from __future__ import annotations

from typing import Sequence

import numpy as np

from servers.shared.schemas import PoseAnalysisRequest, PoseAnalysisResponse, PosePoint


def _to_array(landmarks: Sequence[PosePoint]) -> np.ndarray:
    return np.array([[point.x, point.y, point.z] for point in landmarks], dtype=float)


def calculate_movement_score(current_landmarks: Sequence[PosePoint], previous_landmarks: Sequence[PosePoint]) -> float:
    if not current_landmarks or not previous_landmarks:
        return 0.0
    current_array = _to_array(current_landmarks)
    previous_array = _to_array(previous_landmarks)
    if current_array.shape != previous_array.shape:
        usable_count = min(len(current_landmarks), len(previous_landmarks))
        current_array = current_array[:usable_count]
        previous_array = previous_array[:usable_count]
    diff = np.linalg.norm(current_array - previous_array, axis=1)
    return float(np.sum(diff))


def estimate_met(movement_score: float) -> float:
    if movement_score < 1.2:
        return 1.0
    return 3.0 + min(movement_score * 2, 7.0)


def calculate_calories_burned(met: float, user_weight: float, elapsed_seconds: float) -> float:
    return (met * 3.5 * user_weight / 200) / 60 * elapsed_seconds


def analyze_pose_request(request: PoseAnalysisRequest) -> PoseAnalysisResponse:
    movement_score = calculate_movement_score(request.current_landmarks, request.previous_landmarks)
    current_met = estimate_met(movement_score)
    calories_burned = calculate_calories_burned(current_met, request.user_weight, request.elapsed_seconds)
    return PoseAnalysisResponse(
        movement_score=movement_score,
        current_met=current_met,
        calories_burned=calories_burned,
        landmark_count=len(request.current_landmarks),
    )
