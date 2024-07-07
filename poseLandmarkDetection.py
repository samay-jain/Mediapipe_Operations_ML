import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

model_path = './Models/pose_landmarker_full.task'

# MediaPipe task initialization
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize a global variable to store the annotated frame
annotated_frame = None

# Create a pose landmarker instance with the live stream mode
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame
    if result.pose_landmarks:
        annotated_frame = np.copy(output_image.numpy_view())
        for pose_landmark in result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmark
            ])
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=pose_landmarks_proto,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize Pose Landmarker
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Perform pose detection
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    if annotated_frame is not None:
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Pose Landmark Detection', annotated_frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
