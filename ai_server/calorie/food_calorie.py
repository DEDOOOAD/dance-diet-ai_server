from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
FOOD_DB_PATH = BASE_DIR / "통합 식품영양성분DB.xlsx"
FOOD_NAME_COLUMN = "식품명"
FOOD_CALORIE_COLUMN = "에너지(㎉)"


@lru_cache(maxsize=1)
def load_food_calorie_map() -> dict[str, float]:
    if not FOOD_DB_PATH.exists():
        raise FileNotFoundError(f"Food calorie DB file does not exist: {FOOD_DB_PATH}")

    df = pd.read_excel(FOOD_DB_PATH, header=3)

    if FOOD_NAME_COLUMN not in df.columns:
        raise KeyError(f"Food calorie DB does not have '{FOOD_NAME_COLUMN}' column.")
    if FOOD_CALORIE_COLUMN not in df.columns:
        raise KeyError(f"Food calorie DB does not have '{FOOD_CALORIE_COLUMN}' column.")

    calorie_map: dict[str, float] = {}

    for _, row in df.iterrows():
        raw_name = row[FOOD_NAME_COLUMN]
        raw_calorie = row[FOOD_CALORIE_COLUMN]

        if pd.isna(raw_name) or pd.isna(raw_calorie):
            continue

        name = str(raw_name).strip().lower()
        if not name:
            continue

        calorie_map[name] = float(raw_calorie)

    return calorie_map


def lookup_calories(label: str) -> float:
    calorie_map = load_food_calorie_map()
    return float(calorie_map.get(label.strip().lower(), 0.0))
