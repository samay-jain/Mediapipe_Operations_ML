from mediapipe.tasks import python
from mediapipe.tasks.python import text

model_path = './Models/language_detector.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)
options = text.LanguageDetectorOptions(base_options=base_options)
detector = text.LanguageDetector.create_from_options(options)

INPUT_TEXT = "Mera naam Samay Jain hai. tum Balikavadhu dekhna kam karo"

detection_result = detector.detect(INPUT_TEXT)

#displaying scores
for detection in detection_result.detections:
  print(f'{detection.language_code}: ({detection.probability:.2f})')




