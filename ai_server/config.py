import os
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


HOST = "0.0.0.0"
PORT = 8001
APP_NAME = "Ai_server"

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE_PATH = BASE_DIR.parent / ".env"
load_env_file(ENV_FILE_PATH)

POSE_MODEL_PATH = BASE_DIR / "models/pose_landmarker_lite.task"
FOOD_SEGMENTER_MODEL_PATH = BASE_DIR / "models/food_segmenter.pt"
FOOD_CLASSIFIER_MODEL_PATH = BASE_DIR / "models/100c_best.pt"
FOOD_DB_PATH = BASE_DIR / "data/food_db.csv"
FOOD_CLASSIFIER_TOP_K = 5
FOOD_CLASSIFIER_MIN_CONFIDENCE = 0.10
FOOD_VISION_PROVIDER = os.getenv("FOOD_VISION_PROVIDER", "gemini").strip().lower()
FOOD_VISION_TIMEOUT_SECONDS = float(os.getenv("FOOD_VISION_TIMEOUT_SECONDS", "20"))
OPENAI_FOOD_VISION_MODEL = os.getenv("OPENAI_FOOD_VISION_MODEL", "gpt-4.1-mini")
GEMINI_FOOD_VISION_MODEL = os.getenv("GEMINI_FOOD_VISION_MODEL", "gemini-2.5-flash-lite")
