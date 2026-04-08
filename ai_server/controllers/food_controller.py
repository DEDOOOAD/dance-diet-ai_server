from __future__ import annotations

from fastapi import APIRouter
from schemas import FoodAnalysisRequest, FoodAnalysisResponse
from ai_server.services.food_service import analyze_food
import base64

router = APIRouter(prefix="/food", tags=["food"])

@router.post("/analyze", response_model=FoodAnalysisResponse)
def food_analyze(request: FoodAnalysisRequest) -> FoodAnalysisResponse:
    jpg_bytes = base64.b64decode(request.image_base64)
    
    if jpg_bytes is not None:
        print("이미지 정상 수신")

    return analyze_food(request.uuid, jpg_bytes)