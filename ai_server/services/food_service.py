from __future__ import annotations

import logging
from typing import Any

from schemas import FoodAnalysisResponse, FoodItem

from ai_server.calorie.food_calorie import lookup_calories
from ai_server.food_ai.food_classifier import classify_food_image
from ai_server.food_ai.food_segmenter import run_segmentation_model

logger = logging.getLogger(__name__)


def build_food_result(label: str, confidence: float, calories: float) -> FoodItem:
    return FoodItem(label=label, calories=calories, confidence=confidence)

def assemble_foods(
    segments: list[bytes],
    classifier_model: Any,
    calorie_map: dict[str, float],
) -> list[FoodItem]:
    foods: list[FoodItem] = []

    for segment_bytes in segments:
        label, confidence = classify_food_image(segment_bytes, classifier_model)
        if label == "unknown":
            logger.warning("food classification returned unknown")

        calories = lookup_calories(label, calorie_map)
        if calories == 0.0:
            logger.warning("calorie lookup returned 0.0 for label=%s", label)

        foods.append(build_food_result(label, confidence, calories))

    return foods

def analyze_food(
    uuid: str,
    jpg_bytes: bytes,
    segment_model: Any,
    classifier_model: Any,
    calorie_map: dict[str, float],
) -> FoodAnalysisResponse:
    segments = run_segmentation_model(jpg_bytes, segment_model)
    if not segments:
        logger.warning("food segmentation returned no segments for uuid=%s", uuid)

    foods = assemble_foods(segments, classifier_model, calorie_map)
    logger.info("food analysis completed for uuid=%s food_count=%s", uuid, len(foods))

    return FoodAnalysisResponse(uuid=uuid, foods=foods)

    # 테스트용 목업 데이터
    # test_food = {
    #     "uuid": uuid,
    #     "foods": 
    #     [
    #         {
    #             "label": "바나나",
    #             "calories": 105.0,
    #             "confidence": 0.96
    #         },
    #         {
    #             "label": "닭가슴살",
    #             "calories": 165.0,
    #             "confidence": 0.88
    #         }
    #     ]
    # }
    # return test_food
