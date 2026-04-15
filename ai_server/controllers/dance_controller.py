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

    print(f"[{session_id}] 메인 서버와 연결되었습니다.")

    try:
        while True:
            payload = await websocket.receive_json()

            # 테스트용
            message_type = payload.get("type")

            if message_type == "ping":
                await websocket.send_json(
                    {
                        "type": "ai_ready",
                        "session_id": session_id,
                    }
                )
                await websocket.close()
                return

            if message_type == "frame_base64":
                try:
                    frame_bytes = base64.b64decode(payload.get("image_base64", ""))
                except Exception:
                    frame_bytes = b""

                result = await analyze_dance(session_id, frame_bytes)

                if result is None:
                    continue

                if hasattr(result, "model_dump"):
                    await websocket.send_json(result.model_dump(mode="json"))
                else:
                    await websocket.send_json(result)



            # if not isinstance(payload, dict):
            #     await websocket.send_json(
            #         {
            #             "type": "error",
            #             "message": "payload가 잘못되었습니다.",
            #         }
            #     )
            #     continue

            # payload_session_id = payload.get("session_id")
            # if payload_session_id != session_id:
            #     await websocket.send_json(
            #         {
            #             "type": "error",
            #             "message": "session_id 불일치.",
            #             "path_session_id": session_id,
            #             "payload_session_id": payload_session_id,
            #         }
            #     )
            #     continue

            # message_type = payload.get("type")

            # if message_type == "ping":
            #     await websocket.send_json(
            #         {
            #             "type": "ai_ready",
            #             "session_id": session_id,
            #         }
            #     )
            #     await websocket.close()
            #     return

            # if message_type != "frame_base64":
            #     await websocket.send_json(
            #         {
            #             "type": "error",
            #             "message": "type 불일치.",
            #             "expected_type": "frame_base64",
            #             "received_type": message_type,
            #         }
            #     )
            #     continue

            # try:
            #     frame_message = LiveFrameMessage.model_validate(payload)
            # except ValidationError as exc:
            #     await websocket.send_json(
            #         {
            #             "type": "error",
            #             "message": "LiveFrameMessage payload가 잘못되었습니다.",
            #             "detail": exc.errors(),
            #         }
            #     )
            #     continue

            # try:
            #     frame_bytes = base64.b64decode(
            #         frame_message.get_frame_data(),
            #         validate=True,
            #     )
            # except (ValueError, binascii.Error) as exc:
            #     await websocket.send_json(
            #         {
            #             "type": "error",
            #             "message": "image_base64 payload가 잘못되었습니다.",
            #             "detail": str(exc),
            #         }
            #     )
            #     continue

            # result = await analyze_dance(frame_message.session_id, frame_bytes)
            # if result is None:
            #     continue

            # if hasattr(result, "model_dump"):
            #     await websocket.send_json(result.model_dump(mode="json"))
            # else:
            #     await websocket.send_json(result)

    except WebSocketDisconnect:
        print(f"[{session_id}] 메인 서버와 연결이 종료되었습니다.")

    except Exception as e:
        print(f"[{session_id}] error: {e}")
        await websocket.close()

    finally:
        clear_session(session_id)
