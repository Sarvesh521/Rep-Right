import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification

class_names =['bicep_curl','pushup','shoulder press','squat']

processor = AutoImageProcessor.from_pretrained("siddhantuniyal/exercise-detection")
model = AutoModelForImageClassification.from_pretrained("siddhantuniyal/exercise-detection")
model.eval()

def preprocess_frame(frame, target_size=(224, 224)):
    img = cv2.resize(frame, target_size)
    img_array = np.array(img) / 255.0 
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0) 
    return img_tensor

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            processed_frame = preprocess_frame(frame)
            with torch.no_grad():
                output = model(processed_frame)
                logits = output.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
            
            print(f"Frame {frame_count}: {class_names[predicted_class.item()]} ({probabilities[0][predicted_class.item()]:.2f} probability)")
            cv2.imshow('Video Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

video_path = './Model_Classify_Test/Ayush.mp4' 
predict_video(video_path)