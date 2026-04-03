import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# --- 사용자 설정 ---
USER_WEIGHT = 60.0 
total_calories = 0.0
prev_landmarks_array = None 
last_time = time.time()
movement_score = 0.0 # 초기화

def calculate_movement(current_landmarks, prev_landmarks_array):
    """현재 랜드마크와 이전 랜드마크 배열의 차이를 계산"""
    current_array = np.array([[l.x, l.y, l.z] for l in current_landmarks])
    
    if prev_landmarks_array is None:
        return 0, current_array
    
    # 모든 관절의 이동 거리 계산
    diff = np.linalg.norm(current_array - prev_landmarks_array, axis=1)
    movement = np.sum(diff)
    
    return movement, current_array

def draw_landmarks_on_image(rgb_image, detection_result):
    """화면에 뼈대와 관절을 그리는 함수"""
    if not detection_result.pose_landmarks:
        return rgb_image
        
    annotated_image = np.copy(rgb_image)
    
    # 그리기 스타일 설정
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)
            
    return annotated_image

# --- 실행부 ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task') 
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO)

with vision.PoseLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    
    print("캠이 시작되었습니다. 종료하려면 'q'를 누르세요.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. 시간 및 프레임 전처리
        current_real_time = time.time()
        elapsed_time = current_real_time - last_time
        last_time = current_real_time

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 2. 포즈 인식
        detection_result = detector.detect_for_video(mp_image, int(current_real_time * 1000))

        # 3. 칼로리 계산 로직
        if detection_result.pose_landmarks:
            current_landmarks = detection_result.pose_landmarks[0]
            movement_score, prev_landmarks_array = calculate_movement(current_landmarks, prev_landmarks_array)
            
            # 임계값 1.2를 넘어야 움직임으로 간주 (가만히 있을 때 상승 방지)
            if movement_score < 1.2: 
                current_met = 1.0 
            else:
                current_met = 3.0 + min(movement_score * 2, 7.0) 

            calories_burned = (current_met * 3.5 * USER_WEIGHT / 200) / 60 * elapsed_time
            total_calories += calories_burned

        # 4. 시각화 (뼈대 그리기)
        annotated_rgb_image = draw_landmarks_on_image(rgb_frame, detection_result)
        # OpenCV 출력을 위해 다시 BGR로 변환
        display_frame = cv2.cvtColor(annotated_rgb_image, cv2.COLOR_RGB2BGR)

        # 5. 텍스트 정보 표시
        cv2.putText(display_frame, f"Score: {movement_score:.2f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Total: {total_calories:.2f} kcal", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        cv2.imshow('Dance Diet Tracker', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()