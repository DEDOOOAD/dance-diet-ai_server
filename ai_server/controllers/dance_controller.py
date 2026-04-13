from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ai_server.services.dance_service import analyze_dance, clear_session

router = APIRouter(prefix="/dance", tags=["dance"])

# @router.post("/analyze", response_model=PoseAnalysisResponse)
# async def dance_analyze(request: PoseAnalysisRequest) -> PoseAnalysisResponse:
#     return await analyze_dance(request)

@router.websocket("/analyze/{session_id}")
async def dance_analyze(websocket: WebSocket, session_id: str):
    stream_type = websocket.headers.get("x-stream-type")

    if stream_type != "frame":
        await websocket.close()
        return

    await websocket.accept()

    print(f"[{session_id}] 메인 서버와 연결되었습니다.")

    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            result = await analyze_dance(session_id, frame_bytes)
            if result is not None:
                await websocket.send_json(result.model_dump())
    
    except WebSocketDisconnect:
        print(f"[{session_id}] 메인 서버와 연결이 종료되었습니다.")
        await websocket.close()

    except Exception as e:
        print(f"[{session_id}] 에러: {e}")
        await websocket.close()

    finally:
        await clear_session(session_id)
