from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
import cv2
import numpy as np
from ai_server.dance_ai.pose_analyzer import analyze_pose_frame, clear_pose_session
from schemas import AiLiveAnalysisMessage

_SESSION_STATES: dict[str, dict[str, Any]] = {}
logger = logging.getLogger(__name__)


def clear_session(session_id: str) -> None:
    _SESSION_STATES.pop(session_id, None)
    clear_pose_session(session_id)
    logger.info("[%s] dance analysis session cleared", session_id)

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
    pose_detector: Any,
    frame_index: int | None = None,
    user_weight: float | None = None,
    user_height: float | None = None,
) -> list[AiLiveAnalysisMessage]:
    if frame_bytes is None:
        logger.debug("[%s] empty frame bytes", session_id)
        return []

    frame = decode_frame(frame_bytes)
    if frame is None or frame.size == 0:
        logger.debug("[%s] failed to decode frame", session_id)
        return []

    logger.info(
        "[%s] analysis frame ready: frame_index=%s shape=%s",
        session_id,
        frame_index,
        frame.shape,
    )

    if frame_index is None:
        return [
            _analyze_ordered_frame(
                session_id,
                frame,
                pose_detector,
                None,
                user_weight,
                user_height,
            )
        ]

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
        logger.debug(
            "[%s] skipped stale frame: frame_index=%s next_frame_index=%s",
            session_id,
            frame_index,
            next_frame_index,
        )
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
                pose_detector,
                ordered_index,
                user_weight,
                user_height,
            )
        )
        state["next_frame_index"] = ordered_index + 1

    if not results:
        logger.info(
            "[%s] frame queued: frame_index=%s waiting_for=%s pending_count=%s",
            session_id,
            frame_index,
            state["next_frame_index"],
            len(pending_frames),
        )

    return results


def _analyze_ordered_frame(
    session_id: str,
    frame: Any,
    pose_detector: Any,
    frame_index: int | None,
    user_weight: float | None,
    user_height: float | None,
) -> AiLiveAnalysisMessage:
    analysis_result = analyze_pose_frame(
        frame,
        detector=pose_detector,
        session_id=session_id,
        frame_index=frame_index,
        user_weight=user_weight,
        user_height=user_height,
    )
    return AiLiveAnalysisMessage(
        session_id=session_id,
        processed_at=datetime.now(),
        calories_burned=analysis_result["calories_burned"],
        movement_score=analysis_result["movement_score"],
    )
