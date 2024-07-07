import cv2
import mediapipe as mp
import numpy as np

model_path = './Models/hand_landmarker.task'

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize drawing utilities from MediaPipe
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.flip(rgb_frame, 1)
    results = hands.process(rgb_frame)
    rgb_frame = cv2.flip(rgb_frame, 1)

    frame = cv2.flip(frame, 1)
    # Draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    cv2.imshow('MediaPipe Hand Landmarker Detection', cv2.flip(processed_frame, 1))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
