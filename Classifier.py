import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pyttsx3
from moviepy.editor import VideoFileClip

class_names = ['bicep_curl', 'pushup', 'shoulder press', 'squat']

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

def predict_video(video_path=None):
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_count = 0
    audio_output=[]
    flag=0
    while cap.isOpened():
        if flag==0 and len(audio_output)>50:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1)
            text = f"{max((audio_output), key = audio_output.count)} Exercise Detected"
            engine.say(text)
            engine.runAndWait()
            flag=1
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            processed_frame = preprocess_frame(frame)
            with torch.no_grad():
                output = model(processed_frame)
                logits = output.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
            
            prediction_text = f"{class_names[predicted_class.item()]} ({probabilities[0][predicted_class.item()]:.2f})"
            print(f"Frame {frame_count}: {prediction_text}")
            audio_output.append(class_names[predicted_class.item()])
            
            # Write the prediction onto the frame
            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Prediction', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    if flag==0:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        text = f"{max(set(audio_output), key = audio_output.count)} Exercise Detected"
        engine.say(text)
        engine.runAndWait()

predict_video('./Model_Classify_Test/pushupv1.mp4')