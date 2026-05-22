from pathlib import Path

HOST = "0.0.0.0"
PORT = 8001
APP_NAME = "Ai_server"

BASE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = BASE_DIR / "models/pose_landmarker_lite.task"
FOOD_SEGMENTER_MODEL_PATH = BASE_DIR / "models/food_segmenter.pt"
FOOD_CLASSIFIER_MODEL_PATH = BASE_DIR / "models/100c_best.pt"
FOOD_DB_PATH = BASE_DIR / "data/food_db.csv"