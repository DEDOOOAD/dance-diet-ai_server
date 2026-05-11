from __future__ import annotations

import cv2
import numpy as np
from typing import Any
from datetime import datetime
from ai_server.dance_ai.pose_analyzer import analyze_pose_frame
from schemas import AiLiveAnalysisMessage

_SESSION_STATES: dict[str, dict[str, Any]] = {}


def clear_session(session_id: str) -> None:
    _SESSION_STATES.pop(session_id, None)

def decode_frame(frame_bytes: bytes):
    encoded_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    if encoded_frame.size == 0:
        return None

    frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    
    return frame

async def analyze_dance(session_id: str, frame_bytes: bytes) -> AiLiveAnalysisMessage | None:
    if frame_bytes is None:
        return None

    frame = decode_frame(frame_bytes)
    if frame is None or frame.size == 0:
        return None

    analysis_result = await analyze_pose_frame(frame)

    return AiLiveAnalysisMessage(
        session_id=session_id,
        processed_at=datetime.now(),
        calories_burned=analysis_result["calories_burned"],
        movement_score=analysis_result["movement_score"],
    )
