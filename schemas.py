from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field




class ServerInfo(BaseModel):
    name: str
    role: str
    status: str = "ok"

class UserSignUp(BaseModel):
    name: str
    email: str
    password: str
    age: int
    uuid: str 

class GeneralAiProxyResponse(BaseModel):
    ai_server_url: str
    recommended_endpoints: list[str]
    note: str


class ToggleUpdateRequest(BaseModel):
    key: str
    value: bool


class ToggleUpdateResponse(BaseModel):
    message: str
    toggles: dict[str, bool]


class LiveSessionStartRequest(BaseModel):
    user_id: str
    dance_type: str | None = None
    content_id: str | None = None


class LiveSessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    started_at: datetime
    ws_url: str
    dance_type: str | None = None
    content_id: str | None = None


class LiveSessionEndRequest(BaseModel):
    session_id: str


class LiveSessionEndResponse(BaseModel):
    session_id: str
    status: str
    ended_at: datetime
    total_frames: int
    total_calories: float
    message: str


class PosePoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class PoseAnalysisRequest(BaseModel):
    current_landmarks: list[PosePoint] = Field(default_factory=list)
    previous_landmarks: list[PosePoint] = Field(default_factory=list)
    user_weight: float = 60.0
    elapsed_seconds: float = 1.0


class PoseAnalysisResponse(BaseModel):
    movement_score: float
    current_met: float
    calories_burned: float
    landmark_count: int


class GifAnalysisRequest(BaseModel):
    gif_path: str = "test.gif"
    max_frames: int | None = None


class GifAnalysisResponse(BaseModel):
    gif_path: str
    processed_frames: int
    detected_frames: int


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
