from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from schemas import FoodAnalysisResponse, FoodItem

from ai_server.calorie.food_calorie import lookup_calories
from ai_server.food_ai.food_classifier import classify_food_candidates
from ai_server.food_ai.food_vision_fallback import FoodVisionFallbackAnalyzer

logger = logging.getLogger(__name__)


def build_food_result(label: str, confidence: float, calories: float) -> FoodItem:
    return FoodItem(label=label, calories=calories, confidence=confidence)


def assemble_foods(
    jpg_bytes: bytes,
    classifier_model: Any,
    calorie_map: dict[str, float],
) -> list[FoodItem]:
    predictions = classify_food_candidates(jpg_bytes, classifier_model)
    fallback_predictions = FoodVisionFallbackAnalyzer().analyze(jpg_bytes)
    if fallback_predictions:
        logger.info(
            "food vision fallback verified food_count=%s classifier_candidate_count=%s",
            len(fallback_predictions),
            len(predictions),
        )
        return [
            build_food_result(
                prediction.label,
                prediction.confidence,
                resolve_calories(prediction.label, prediction.calories, calorie_map),
            )
            for prediction in fallback_predictions
        ]

    foods: list[FoodItem] = []
    for candidate_index, (label, confidence) in enumerate(predictions, start=1):
        logger.info(
            "food classification candidate: candidate=%s label=%s confidence=%.6f",
            candidate_index,
            label,
            confidence,
        )

        if label == "unknown":
            logger.warning(
                "food classification returned unknown: candidate=%s confidence=%.6f",
                candidate_index,
                confidence,
            )

        foods.append(build_food_result(label, confidence, resolve_calories(label, 0.0, calorie_map)))

    return foods


def resolve_calories(label: str, fallback_calories: float, calorie_map: dict[str, float]) -> float:
    if fallback_calories > 0.0:
        return fallback_calories

    calories = lookup_calories(label, calorie_map)
    if calories == 0.0:
        logger.warning("calorie lookup returned 0.0 for label=%s", label)

    return calories


def analyze_food(
    uuid: str,
    jpg_bytes: bytes,
    image_filename: str | None,
    classifier_model: Any,
    calorie_map: dict[str, float],
) -> FoodAnalysisResponse:
    foods = assemble_foods(jpg_bytes, classifier_model, calorie_map)
    total_calories = round(sum(food.calories for food in foods), 1)
    logger.info(
        "food analysis completed for uuid=%s food_count=%s total_calories=%.1f",
        uuid,
        len(foods),
        total_calories,
    )

    return FoodAnalysisResponse(
        uuid=uuid,
        foods=foods,
        total_calories=total_calories,
        image_filename=image_filename,
        source="ai-food-server",
        analyzed_at=datetime.now().astimezone(),
    )

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
