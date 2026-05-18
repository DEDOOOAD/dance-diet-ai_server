from __future__ import annotations

import base64
import binascii
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from ai_server.services.dance_service import analyze_dance, clear_session
from schemas import LiveFrameMessage

router = APIRouter(prefix="/ws/dance", tags=["dance"])


@router.websocket("/analyze/{session_id}")
async def dance_analyze(websocket: WebSocket, session_id: str):
    await websocket.accept()

    print(f"[{session_id}] main server connected")

    try:
        while True:
            payload = await websocket.receive_json()

            if not isinstance(payload, dict):
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
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Invalid LiveFrameMessage payload.",
                            "detail": exc.errors(),
                        }
                    )
                    continue

                if frame_message.session_id != session_id:
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
                    frame_index=frame_message.frame_index,
                    user_weight=frame_message.user_weight,
                )

                if not results:
                    continue

                for result in results:
                    if hasattr(result, "model_dump"):
                        await websocket.send_json(result.model_dump(mode="json"))
                    else:
                        await websocket.send_json(result)

    except WebSocketDisconnect:
        print(f"[{session_id}] main server disconnected")

    except Exception as e:
        print(f"[{session_id}] error: {e}")
        await websocket.close()

    finally:
        clear_session(session_id)
