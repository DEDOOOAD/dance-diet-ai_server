from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from schemas import PoseAnalysisRequest, PoseAnalysisResponse
from ai_server.services.dance_service import analyze_dance

router = APIRouter(prefix="/dance", tags=["dance"])

# @router.post("/analyze", response_model=PoseAnalysisResponse)
# async def dance_analyze(request: PoseAnalysisRequest) -> PoseAnalysisResponse:
#     return await analyze_dance(request)

@router.websocket("/analyze")
async def dance_analyze(websocket: WebSocket):
    await websocket.accept()
    print("메인 서버와 연결되었습니다.")

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                print(f"메인 서버로부터 영상 데이터 수신: {len(message['bytes'])} bytes")

            result = await analyze_dance(message)
            if result is not None:
                await websocket.send_json(result)
    
    except WebSocketDisconnect:
        print("메인 서버와 연결이 종료되었습니다.")
