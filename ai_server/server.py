from __future__ import annotations

from fastapi import FastAPI
import base64

from ai_server.config import APP_NAME
from ai_server.services.gif_pose_service import analyze_gif_file
from ai_server.services.pose_metrics import analyze_pose_request
from schemas import GifAnalysisRequest, GifAnalysisResponse, PoseAnalysisRequest, PoseAnalysisResponse, ServerInfo, FoodAnalysisResponse, FoodAnalysisRequest
from ai_server.services.food_pipeline import analyze_food_pipeline

app = FastAPI(
    title="AI Analysis API",
    version="1.0.0",
    description="AI server for pose and GIF analysis.",
)

@app.post("/analyze/pose", response_model=PoseAnalysisResponse)
def analyze_pose(request: PoseAnalysisRequest) -> PoseAnalysisResponse:
    return analyze_pose_request(request)

@app.post("/food/analyze", response_model=FoodAnalysisResponse)
async def food_analyze(request: FoodAnalysisRequest) -> FoodAnalysisResponse:
    jpg_bytes = base64.b64decode(request.image_base64)
    return await analyze_food_pipeline(request.uuid, jpg_bytes)
