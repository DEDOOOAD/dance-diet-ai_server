from __future__ import annotations

from typing import Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, model_validator


class LiveFrameMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["frame_base64"] = "frame_base64"
    UUID: str 
    session_id: str
    frame_index: int = Field(ge=0)
    total_frame: int = Field(ge=0)
    image: str
    user_weight: float | None = Field(default=None, gt=0)
    user_height: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_frame_payload(self) -> "LiveFrameMessage":
        if not self.image:
            raise ValueError("One of image is required.")
        return self

    def get_frame_data(self) -> str:
        if self.image:
            return self.image

        raise ValueError("Frame payload is missing.")


class AiLiveAnalysisMessage(BaseModel):
    type: Literal["ai_analysis"] = "ai_analysis"
    session_id: str
    processed_at: datetime
    calories_burned: float = 0.0
    movement_score: float = 0.0



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
