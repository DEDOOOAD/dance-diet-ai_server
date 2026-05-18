from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
HOST = "0.0.0.0"
PORT = 8001
APP_NAME = "Ai_server"
POSE_MODEL_PATH = BASE_DIR / "models/pose_landmarker_lite.task"
FOOD_SEGMENTER_MODEL_PATH = BASE_DIR / "models/food_segmenter.pt"
FOOD_CLASSIFIER_MODEL_PATH = BASE_DIR / "models/food_classifier.pt"