from __future__ import annotations

import numpy as np
import cv2


def analyze_pose_frame(frame: np.ndarray) -> dict[str, float | int]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

    return {
        "movement_score": movement_score,
        "current_met": current_met,
        "calories_burned": calories_burned,
        "landmark_count": landmark_count,
    }
