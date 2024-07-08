#Face Detection and Hand Landmark Detection


# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# # Initialize MediaPipe Face Detection module
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# # Initialize drawing utilities from MediaPipe
# mp_drawing = mp.solutions.drawing_utils

# def process_frame(frame):
#     # Convert the frame to RGB and flip for selfie view
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb_frame = cv2.flip(rgb_frame, 1)
    
#     # Process hand landmarks
#     hand_results = hands.process(rgb_frame)
    
#     # Process face detection
#     face_results = face_detection.process(rgb_frame)
    
#     # Flip back the RGB frame
#     rgb_frame = cv2.flip(rgb_frame, 1)

#     frame = cv2.flip(frame, 1)
    
#     # Draw hand landmarks on the frame
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#     # Draw face detections on the frame
#     if face_results.detections:
#         for detection in face_results.detections:
#             mp_drawing.draw_detection(frame, detection)
    
#     return frame

# # Main loop to capture from webcam and process frames
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     processed_frame = process_frame(frame)
#     cv2.imshow('MediaPipe Hand Landmarker and Face Detection', cv2.flip(processed_frame, 1))
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#_________________________________________________________________________________________________________________
#Face Detection, Hand Landmark Detection, Face Mesh Detection

# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe modules
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize drawing utilities from MediaPipe
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# def process_frame(frame):
#     # Convert the frame to RGB and flip for selfie view
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     rgb_frame = cv2.flip(rgb_frame, 1)
    
#     # Process hand landmarks
#     hand_results = hands.process(rgb_frame)
    
#     # Process face detection
#     face_results = face_detection.process(rgb_frame)
    
#     # Process face mesh
#     mesh_results = face_mesh.process(rgb_frame)
    
#     # Flip back the RGB frame
#     rgb_frame = cv2.flip(rgb_frame, 1)

#     frame = cv2.flip(frame, 1)
    
#     # Draw hand landmarks on the frame
#     if hand_results.multi_hand_landmarks:
#         for hand_landmarks in hand_results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
#     # Draw face detections on the frame
#     if face_results.detections:
#         for detection in face_results.detections:
#             mp_drawing.draw_detection(frame, detection)

#     # Draw face mesh on the frame
#     if mesh_results.multi_face_landmarks:
#         for face_landmarks in mesh_results.multi_face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
#             mp_drawing.draw_landmarks(
#                 image=frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_IRISES,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    
#     return frame

# # Main loop to capture from webcam and process frames
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     processed_frame = process_frame(frame)
#     cv2.imshow('MediaPipe Hand Landmarker, Face Detection, and Face Mesh', cv2.flip(processed_frame, 1))
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





#_________________________________________________________________________________________________________________
#Face Detection, Hand Landmark Detection, Face Mesh Detection, Skeleton/Pose Detection

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# MediaPipe modules for face detection, face mesh, hand detection, and pose landmarking
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Initialize MediaPipe modules
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

# Initialize drawing utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variable to store annotated frame
annotated_frame = None

# Pose Landmarker callback function
def print_result(result: mp.solutions.pose.PoseLandmarks, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame
    if result.pose_landmarks:
        annotated_frame = np.copy(output_image.numpy_view())
        for pose_landmark in result.pose_landmarks.landmark:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmark
            ])
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=pose_landmarks_proto,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

# Pose Landmarker options and initialization
model_path = './Models/pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

landmarker = PoseLandmarker.create_from_options(options)

# Main loop to capture from webcam and process frames
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Process hand landmarks
    hand_results = hands.process(mp_image)

    # Process face detection
    face_results = face_detection.process(mp_image)

    # Process face mesh
    mesh_results = face_mesh.process(mp_image)

    # Perform pose detection
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    # Draw hand landmarks on the frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw face detections on the frame
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Draw face mesh on the frame
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # Display pose landmarking results if available
    if annotated_frame is not None:
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Pose Landmark Detection', annotated_frame_bgr)
    else:
        cv2.imshow('MediaPipe Hand Landmarker, Face Detection, and Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()










#_________________________________________________________________________________________________________________
#Face Detection, Hand Landmark Detection, Face Mesh Detection, live Gesture Recognition

