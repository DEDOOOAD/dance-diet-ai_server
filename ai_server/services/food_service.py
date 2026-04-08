from __future__ import annotations

from ai_server.food_ai.food_classifier import classify_food_image
from ai_server.food_ai.food_segmenter import run_segmentation_model
from ai_server.calorie.food_calorie import lookup_calories
from schemas import FoodAnalysisResponse, FoodItem


def build_food_result(label: str, confidence: float, calories: float) -> FoodItem:
    return FoodItem(label=label, calories=calories, confidence=confidence)

def assemble_foods(segments: list[bytes]) -> list[FoodItem]:
    foods: list[FoodItem] = []

    for segment_bytes in segments:
        label, confidence = classify_food_image(segment_bytes)
        calories = lookup_calories(label)
        foods.append(build_food_result(label, confidence, calories))

    return foods

def analyze_food(uuid: str, jpg_bytes: bytes) -> FoodAnalysisResponse:
    segments = run_segmentation_model(jpg_bytes)
    foods = assemble_foods(segments)
    return FoodAnalysisResponse(uuid=uuid, foods=foods)
