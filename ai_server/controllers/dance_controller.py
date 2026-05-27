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
    await websocket.send_json({"type": "ai_ready"})

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

                logger.info(
                    "[%s] frame received: frame_index=%s total_frame=%s",
                    session_id,
                    frame_message.frame_index,
                    frame_message.total_frame,
                )

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

                logger.info(
                    "[%s] frame decoded: frame_index=%s image_bytes=%s",
                    session_id,
                    frame_message.frame_index,
                    len(frame_bytes),
                )

                results = await analyze_dance(
                    session_id=frame_message.session_id,
                    frame_bytes=frame_bytes,
                    pose_detector=pose_detector,
                    frame_index=frame_message.frame_index,
                    user_weight=frame_message.user_weight,
                    user_height=frame_message.user_height,
                )

                if not results:
                    logger.info(
                        "[%s] analysis pending: frame_index=%s no result to send",
                        session_id,
                        frame_message.frame_index,
                    )
                    continue

                logger.info(
                    "[%s] analysis completed: frame_index=%s result_count=%s",
                    session_id,
                    frame_message.frame_index,
                    len(results),
                )

                for result in results:
                    if hasattr(result, "model_dump"):
                        result_payload = result.model_dump(mode="json")
                    else:
                        result_payload = result

                    await websocket.send_json(result_payload)

                    logger.info(
                        "[%s] analysis result sent: calories_burned=%.6f movement_score=%.6f",
                        session_id,
                        result_payload.get("calories_burned", 0.0),
                        result_payload.get("movement_score", 0.0),
                    )

    except WebSocketDisconnect as exc:
        logger.info("[%s] main server disconnected: close_code=%s", session_id, exc.code)

    except Exception:
        logger.exception("[%s] unexpected error", session_id)
        await websocket.close()

    finally:
        if hasattr(pose_detector, "close"):
            pose_detector.close()
        websocket.app.state.pose_detectors.pop(session_id, None)
        clear_session(session_id)
