from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from servers.ai_server.config import POSE_MODEL_PATH
from servers.shared.schemas import GifAnalysisRequest, GifAnalysisResponse


def analyze_gif_file(request: GifAnalysisRequest) -> GifAnalysisResponse:
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)

    processed_frames = 0
    detected_frames = 0
    with vision.PoseLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(request.gif_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            processed_frames += 1
            if detection_result.pose_landmarks:
                detected_frames += 1
            if request.max_frames is not None and processed_frames >= request.max_frames:
                break
        cap.release()
    return GifAnalysisResponse(gif_path=request.gif_path, processed_frames=processed_frames, detected_frames=detected_frames)
