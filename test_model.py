from vosk import Model
import os

MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    print("Model path does not exist!")
else:
    try:
        model = Model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
