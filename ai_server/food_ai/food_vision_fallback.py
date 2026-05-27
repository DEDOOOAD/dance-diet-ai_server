from __future__ import annotations

import base64
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_server.config import (
    FOOD_VISION_PROVIDER,
    FOOD_VISION_TIMEOUT_SECONDS,
    GEMINI_FOOD_VISION_MODEL,
    OPENAI_FOOD_VISION_MODEL,
)

logger = logging.getLogger(__name__)

FOOD_ANALYSIS_PROMPT = """
Analyze this food image.

Return only valid JSON in this exact shape:
{
  "foods": [
    {
      "label": "food name",
      "calories": 0,
      "confidence": 0.0
    }
  ]
}

Rules:
- Detect all visible edible food items, especially Korean foods.
- The label value must be a Korean food name written in Korean.
- Do not return English, romanized Korean, or mixed-language food labels.
- Estimate calories for the visible serving size, not per 100g.
- confidence must be between 0 and 1.
- Do not include utensils, bowls, plates, tables, packaging, or non-food objects.
- If no food is visible, return {"foods":[]}.
- The JSON must match the server FoodItem schema fields: label, calories, confidence.
""".strip()

FOOD_ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "foods": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "calories": {"type": "number", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "calories", "confidence"],
            },
        },
    },
    "required": ["foods"],
}


@dataclass(frozen=True)
class VisionFoodPrediction:
    label: str
    calories: float
    confidence: float


class FoodVisionClient(ABC):
    @abstractmethod
    def analyze(self, jpg_bytes: bytes) -> list[VisionFoodPrediction]:
        raise NotImplementedError


class HttpJsonClient:
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class OpenAIFoodVisionClient(FoodVisionClient):
    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_FOOD_VISION_MODEL,
        timeout_seconds: float = FOOD_VISION_TIMEOUT_SECONDS,
        http_client: HttpJsonClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or HttpJsonClient()

    def analyze(self, jpg_bytes: bytes) -> list[VisionFoodPrediction]:
        image_base64 = base64.b64encode(jpg_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": FOOD_ANALYSIS_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "food_analysis_response",
                    "schema": FOOD_ANALYSIS_RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
        }
        response = self.http_client.post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            self.timeout_seconds,
        )
        return parse_food_predictions(extract_openai_text(response))


class GeminiFoodVisionClient(FoodVisionClient):
    def __init__(
        self,
        api_key: str,
        model: str = GEMINI_FOOD_VISION_MODEL,
        timeout_seconds: float = FOOD_VISION_TIMEOUT_SECONDS,
        http_client: HttpJsonClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client or HttpJsonClient()

    def analyze(self, jpg_bytes: bytes) -> list[VisionFoodPrediction]:
        image_base64 = base64.b64encode(jpg_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64,
                            }
                        },
                        {"text": FOOD_ANALYSIS_PROMPT},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
            },
        }
        response = self.http_client.post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            payload,
            {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            self.timeout_seconds,
        )
        return parse_food_predictions(extract_gemini_text(response))


class FoodVisionFallbackAnalyzer:
    def __init__(self, provider: str = FOOD_VISION_PROVIDER) -> None:
        self.provider = provider

    def analyze(self, jpg_bytes: bytes) -> list[VisionFoodPrediction]:
        client = self._create_client()
        if client is None:
            return []

        try:
            return client.analyze(jpg_bytes)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("food vision fallback failed: provider=%s error=%s", self.provider, exc)
            return []

    def _create_client(self) -> FoodVisionClient | None:
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                logger.info("OpenAI food vision fallback skipped because OPENAI_API_KEY is not set")
                return None

            return OpenAIFoodVisionClient(api_key=api_key)

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                logger.info("Gemini food vision fallback skipped because GEMINI_API_KEY is not set")
                return None

            return GeminiFoodVisionClient(api_key=api_key)

        if self.provider:
            logger.warning("unsupported food vision fallback provider: %s", self.provider)

        return None


def parse_food_predictions(response_text: str) -> list[VisionFoodPrediction]:
    data = json.loads(extract_json_object(response_text))
    raw_foods = data.get("foods", [])
    if not isinstance(raw_foods, list):
        return []

    predictions: list[VisionFoodPrediction] = []
    for raw_food in raw_foods:
        if not isinstance(raw_food, dict):
            continue

        label = str(raw_food.get("label", "")).strip()
        if not label:
            continue

        if not contains_korean(label):
            logger.warning("food vision fallback skipped non-Korean label=%s", label)
            continue

        calories = to_float(raw_food.get("calories"), default=0.0)
        confidence = clamp(to_float(raw_food.get("confidence"), default=0.0), 0.0, 1.0)
        predictions.append(
            VisionFoodPrediction(
                label=label,
                calories=max(calories, 0.0),
                confidence=confidence,
            )
        )

    return predictions


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is None:
        raise ValueError("food vision response does not contain a JSON object")

    return match.group(0)


def extract_openai_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    chunks: list[str] = []
    for output in response.get("output", []):
        for content in output.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "\n".join(chunks)


def extract_gemini_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "\n".join(chunks)


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def contains_korean(value: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in value)
