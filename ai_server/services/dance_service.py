from __future__ import annotations

import cv2
import numpy as np
from typing import Any
from datetime import datetime
from ai_server.dance_ai.pose_analyzer import analyze_pose_frame, clear_pose_session
from schemas import AiLiveAnalysisMessage

_SESSION_STATES: dict[str, dict[str, Any]] = {}


def clear_session(session_id: str) -> None:
    _SESSION_STATES.pop(session_id, None)
    clear_pose_session(session_id)

def decode_frame(frame_bytes: bytes):
    encoded_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    if encoded_frame.size == 0:
        return None

    frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    
    return frame

async def analyze_dance(
    session_id: str,
    frame_bytes: bytes,
    frame_index: int | None = None,
    user_weight: float | None = None,
) -> list[AiLiveAnalysisMessage]:
    if frame_bytes is None:
        return []

    frame = decode_frame(frame_bytes)
    if frame is None or frame.size == 0:
        return []

    if frame_index is None:
        return [_analyze_ordered_frame(session_id, frame, None, user_weight)]

    state = _SESSION_STATES.setdefault(
        session_id,
        {
            "next_frame_index": 0,
            "pending_frames": {},
        },
    )
    pending_frames: dict[int, Any] = state["pending_frames"]

    next_frame_index = state["next_frame_index"]
    if frame_index < next_frame_index:
        return []

    pending_frames[frame_index] = frame

    results: list[AiLiveAnalysisMessage] = []
    while state["next_frame_index"] in pending_frames:
        ordered_index = state["next_frame_index"]
        ordered_frame = pending_frames.pop(ordered_index)
        results.append(
            _analyze_ordered_frame(
                session_id,
                ordered_frame,
                ordered_index,
                user_weight,
            )
        )
        state["next_frame_index"] = ordered_index + 1

    return results


def _analyze_ordered_frame(
    session_id: str,
    frame: Any,
    frame_index: int | None,
    user_weight: float | None,
) -> AiLiveAnalysisMessage:
    analysis_result = analyze_pose_frame(
        frame,
        session_id=session_id,
        frame_index=frame_index,
        user_weight=user_weight,
    )
    return AiLiveAnalysisMessage(
        session_id=session_id,
        processed_at=datetime.now(),
        calories_burned=analysis_result["calories_burned"],
        movement_score=analysis_result["movement_score"],
    )
