from __future__ import annotations

import numpy as np


def analyze_pose_frame(frame: np.ndarray) -> dict[str, float | int]:
    if frame is None or frame.size == 0:
        return {
            "movement_score": 0.0,
            "calories_burned": 0.0,
            "landmark_count": 0,
        }

    return {
        "movement_score": 50.0,
        "calories_burned": 0.0,
        "landmark_count": 0,
    }
