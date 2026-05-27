from __future__ import annotations

from typing import Any
from fastapi import Request


def get_classifier_model(request: Request) -> Any:
    return request.app.state.classifier_model


def get_food_calorie_map(request: Request) -> dict[str, float]:
    return request.app.state.food_calorie_map


def get_pose_detectors(request: Request) -> dict[str, Any]:
    return request.app.state.pose_detectors
