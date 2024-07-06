from mediapipe.tasks import python
from mediapipe.tasks.python import text

model_path = './Models/language_detector.tflite'
# STEP 2: Create a LanguageDetector object.
base_options = python.BaseOptions(model_asset_path=model_path)
options = text.LanguageDetectorOptions(base_options=base_options)
detector = text.LanguageDetector.create_from_options(options)

INPUT_TEXT = "Mera naam Samay Jain hai. tum Balikavadhu dekhna kam karo"

# STEP 3: Get the language detcetion result for the input text.
detection_result = detector.detect(INPUT_TEXT)

# STEP 4: Process the detection result and print the languages detected and
# their scores.

for detection in detection_result.detections:
  print(f'{detection.language_code}: ({detection.probability:.2f})')




