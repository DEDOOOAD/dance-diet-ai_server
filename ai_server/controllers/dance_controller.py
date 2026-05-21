from __future__ import annotations

import base64
import binascii
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from ai_server.dance_ai.pose_analyzer import load_pose_detector
from ai_server.services.dance_service import analyze_dance, clear_session
from schemas import LiveFrameMessage

router = APIRouter(prefix="/ws/dance", tags=["dance"])
logger = logging.getLogger(__name__)


@router.websocket("/analyze/{session_id}")
async def dance_analyze(websocket: WebSocket, session_id: str):
    await websocket.accept()
    previous_detector = websocket.app.state.pose_detectors.pop(session_id, None)
    if previous_detector is not None and hasattr(previous_detector, "close"):
        previous_detector.close()

    pose_detector = load_pose_detector()
    websocket.app.state.pose_detectors[session_id] = pose_detector

    logger.info("[%s] main server connected", session_id)

    try:
        while True:
            payload = await websocket.receive_json()

            if not isinstance(payload, dict):
                logger.warning(
                    "[%s] invalid payload type: %s",
                    session_id,
                    type(payload).__name__,
                )
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid payload.",
                    }
                )
                continue

            message_type = payload.get("type")

            if message_type == "ping":
                await websocket.send_json(
                    {
                        "type": "ai_ready",
                        "session_id": session_id,
                    }
                )
                continue

            if message_type != "frame_base64":
                logger.warning(
                    "[%s] type mismatch: expected=frame_base64 received=%s",
                    session_id,
                    message_type,
                )
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "type mismatch.",
                        "expected_type": "frame_base64",
                        "received_type": message_type,
                    }
                )
                continue

            if message_type == "frame_base64":
                try:
                    frame_message = LiveFrameMessage.model_validate(payload)
                except ValidationError as exc:
                    logger.warning(
                        "[%s] invalid LiveFrameMessage payload: %s",
                        session_id,
                        exc.errors(),
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid LiveFrameMessage payload.",
                            "detail": exc.errors(),
                        }
                    )
                    continue

                if frame_message.session_id != session_id:
                    logger.warning(
                        "[%s] session_id mismatch: payload_session_id=%s",
                        session_id,
                        frame_message.session_id,
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "session_id mismatch.",
                            "path_session_id": session_id,
                            "payload_session_id": frame_message.session_id,
                        }
                    )
                    continue

                try:
                    frame_bytes = base64.b64decode(
                        frame_message.image,
                        validate=True,
                    )
                except (ValueError, TypeError, binascii.Error) as exc:
                    logger.warning(
                        "[%s] invalid image payload: %s",
                        session_id,
                        exc,
                    )
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid image payload.",
                            "detail": str(exc),
                        }
                    )
                    continue

                results = await analyze_dance(
                    session_id=frame_message.session_id,
                    frame_bytes=frame_bytes,
                    pose_detector=pose_detector,
                    frame_index=frame_message.frame_index,
                    user_weight=frame_message.user_weight,
                    user_height=frame_message.user_height,
                )

                if not results:
                    continue

                for result in results:
                    if hasattr(result, "model_dump"):
                        await websocket.send_json(result.model_dump(mode="json"))
                    else:
                        await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("[%s] main server disconnected", session_id)

    except Exception:
        logger.exception("[%s] unexpected error", session_id)
        await websocket.close()

    finally:
        if hasattr(pose_detector, "close"):
            pose_detector.close()
        websocket.app.state.pose_detectors.pop(session_id, None)
        clear_session(session_id)
