from __future__ import annotations

import base64
import binascii
import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from schemas import FoodAnalysisRequest, FoodAnalysisResponse
from ai_server.dependencies import (
    get_classifier_model,
    get_food_calorie_map,
)
from ai_server.services.food_service import analyze_food

router = APIRouter(prefix="/food", tags=["food"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=FoodAnalysisResponse)
def food_analyze(
    request: FoodAnalysisRequest,
    classifier_model: Any = Depends(get_classifier_model),
    calorie_map: dict[str, float] = Depends(get_food_calorie_map),
) -> FoodAnalysisResponse:
    try:
        jpg_bytes = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("invalid image_base64 payload: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid image_base64 payload.",
        ) from exc

    if not jpg_bytes:
        logger.warning("empty image payload")
        raise HTTPException(
            status_code=400,
            detail="Image payload is empty.",
        )

    logger.info(
        "image_base64 decoded successfully: uuid=%s image_bytes=%s",
        request.uuid,
        len(jpg_bytes),
    )

    try:
        return analyze_food(
            request.uuid,
            jpg_bytes,
            request.image_filename,
            classifier_model,
            calorie_map,
        )
    except Exception as exc:
        logger.exception("food analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Food analysis failed.",
        ) from exc
