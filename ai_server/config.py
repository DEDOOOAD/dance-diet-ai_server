from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
HOST = "0.0.0.0"
PORT = 8002
APP_NAME = "Ai_server"
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
