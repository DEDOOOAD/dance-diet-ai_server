from __future__ import annotations

from pydantic import BaseModel, Field


# class PosePoint(BaseModel):
#     x: float
#     y: float
#     z: float = 0.0

# request 수정
# class PoseAnalysisRequest(BaseModel):
#     current_landmarks: list[PosePoint] = Field(default_factory=list)
#     previous_landmarks: list[PosePoint] = Field(default_factory=list)
#     user_weight: float = 60.0
#     elapsed_seconds: float = 1.0

class PoseAnalysisResponse(BaseModel):
    movement_score: float
    current_met: float
    calories_burned: float
    landmark_count: int


class FoodAnalysisRequest(BaseModel):
    uuid: str
    image_base64: str

class FoodItem(BaseModel):
    label: str
    calories: float
    confidence: float

class FoodAnalysisResponse(BaseModel):
    uuid: str
    foods: list[FoodItem] = Field(default_factory=list)
