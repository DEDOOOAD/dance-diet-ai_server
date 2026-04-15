from __future__ import annotations

import datetime
import cv2
import numpy as np
from typing import Any
from ai_server.dance_ai.pose_analyzer import analyze_pose_frame
from schemas import PoseAnalysisResponse

_SESSION_STATES: dict[str, dict[str, Any]] = {}


def clear_session(session_id: str) -> None:
    _SESSION_STATES.pop(session_id, None)

async def analyze_dance(session_id: str, frame_bytes: bytes) -> PoseAnalysisResponse | None:
    if frame_bytes is None:
        return None

    # encoded_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    # if encoded_frame.size == 0:
    #     return None

    # frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    # if frame is None or frame.size == 0:
    #     return None

    # analysis_result = analyze_pose_frame(frame)

    # return PoseAnalysisResponse(
    #     movement_score=analysis_result["movement_score"],
    #     current_met=analysis_result["current_met"],
    #     calories_burned=analysis_result["calories_burned"],
    #     landmark_count=analysis_result["landmark_count"],
    # )

    # 테스트용 목업 데이터
    test_dance = {
        "type": "ai_analysis",
        "session_id": session_id,
        "processed_at": datetime.now(),
        "calories_burned": 1.0,
        "movement_score": 3.0
    }

    return test_dance
