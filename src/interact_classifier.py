"""
Real-Time Hand Gesture Recognition System
=======================================

This module implements a real-time hand gesture recognition system using computer vision 
and machine learning techniques. It provides an interactive interface for capturing and 
classifying hand gestures through a webcam feed.

Project Overview
--------------
Part of the Computer Vision Master's Project at UC3M (Universidad Carlos III de Madrid)
Author: [Your Name]
Date: [Current Date]
Version: 1.0

Main Features
-----------
* Real-time hand landmark detection using MediaPipe
* Support for both Random Forest and EfficientNet-based classification
* Interactive webcam feed with gesture prediction overlay
* Configurable gesture detection parameters
* Visual feedback for detected hand landmarks
* Performance metrics display

Technical Architecture
-------------------
1. Input Processing:
   - Webcam feed capture using OpenCV
   - Hand landmark detection using MediaPipe
   - Feature extraction and normalization

2. Gesture Classification:
   - Random Forest classifier for landmark-based detection
   - EfficientNet deep learning model for image-based detection
   - Real-time prediction and confidence scoring

3. Visualization:
   - OpenCV-based GUI
   - Hand landmark visualization
   - Prediction confidence display
   - Performance metrics overlay

Dependencies
-----------
* OpenCV (cv2) >= 4.5.0: Video capture and image processing
* MediaPipe >= 0.8.9: Hand landmark detection
* NumPy >= 1.19.0: Numerical computations
* PyTorch >= 1.9.0: Deep learning model support
* Pickle: Model serialization
* tqdm: Progress visualization

Input Requirements
----------------
* Webcam access
* Trained model files:
    - Random Forest model (.pkl)
    - EfficientNet model (if using deep learning approach)
* Consistent lighting conditions
* Single hand in frame (current version)

Output
------
* Real-time visualization window showing:
    - Processed webcam feed
    - Detected hand landmarks
    - Predicted gesture class
    - Confidence scores
    - FPS counter

Usage
-----
Run the script directly to start the interactive session:.


Key Controls:
- 'q': Quit the application
- 'c': Capture current frame
- 's': Save current frame
- 'r': Reset metrics

Configuration
------------
Adjustable parameters:
* CONFIDENCE_THRESHOLD: Minimum confidence for valid predictions (0.0 to 1.0)
* MAX_HANDS: Maximum number of hands to detect (1 for single-hand gestures)
* MODEL_PATH: Path to the trained classifier model file
* CAMERA_INDEX: Webcam device index (0 for default)
* MODEL_TYPE: 'ml' for ML model, 'efficientnet' for EfficientNet model

Performance Considerations
-----------------------
* Recommended minimum specifications:
    - CPU: Intel i5 or equivalent
    - RAM: 8GB
    - Webcam: 720p, 30fps
* Performance may vary based on:
    - Lighting conditions
    - Background complexity
    - Hand positioning
    - System resources

Notes
-----
* Current version optimized for single-hand gestures
* Best results achieved with consistent lighting
* Model accuracy depends on training data quality
* Real-time performance may vary by system

References
---------
1. MediaPipe Hand Landmarker: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
2. OpenCV Documentation: https://docs.opencv.org/
3. [Add relevant papers or resources]

"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import pickle
import numpy as np
import argparse
import json
from pathlib import Path

class HandGestureRecognizer:
    def __init__(self, model_path, model_type, class_lookup):
        """
        Initialize the hand gesture recognizer.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ('efficientnet', 'mobilenet', or 'ml')
            class_lookup: Dictionary mapping indices to class labels
        """
        self.model_type = model_type.lower()
        
        # MediaPipe hand landmarker setup
        base_options = python.BaseOptions(model_asset_path='src/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the appropriate model
        self.model = self._load_model(model_path)
        
        # Label mapping
        self.labels_dict = class_lookup
        
        # Setup image transforms based on model type
        if self.model_type == 'ml':
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.Resize((384, 384) if model_type == 'efficientnet' else (224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_model(self, model_path):
        """Load and prepare model based on type."""
        try:
            if self.model_type == 'ml':
                with open(model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # If loaded_data is a dictionary, extract the model
                    if isinstance(loaded_data, dict):
                        if 'model' in loaded_data:
                            return loaded_data['model']
                        else:
                            raise ValueError("ML model dictionary does not contain 'model' key")
                    # If it's already a model object, return it directly
                    return loaded_data
            
            # Para modelos PyTorch
            if self.model_type == 'efficientnet':
                from efficientNet.efficientnet_transferlearning_finetunning import ASLModel
                model = ASLModel(num_classes=29)
            elif self.model_type == 'mobilenet':
                model = MobileNetModel(num_classes=29)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
            
            # Load the saved model weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            
            # Set the model to evaluation mode for inference
            model.eval()
            model.to(self.device)
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _process_landmarks_for_ml(self, landmarks):
        """Process landmarks for ML model."""
        data_aux = []
        x_ = []
        y_ = []
        
        for landmark in landmarks:
            x_.append(landmark.x)
            y_.append(landmark.y)
            
        for landmark in landmarks:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))
            
        return np.asarray([data_aux])

    def _process_image(self, frame, bbox):
        """Process image for deep learning models."""
        if self.model_type == 'ml':
            return None  # ML model uses landmarks directly
            
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            size_diff = abs(width - height)
            
            if width > height:
                y1 = max(0, y1 - size_diff // 2)
                y2 = min(frame.shape[0], y2 + size_diff // 2)
            else:
                x1 = max(0, x1 - size_diff // 2)
                x2 = min(frame.shape[1], x2 + size_diff // 2)
            
            hand_img = frame[y1:y2, x1:x2]
            if hand_img.size == 0:
                return None
                
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(hand_img)
            
            input_tensor = self.transform(pil_image)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            return input_tensor
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict(self, frame):
        """Predict hand gesture from frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return None, None, None, None
            
        landmarks = detection_result.hand_landmarks[0]
        
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in landmarks]
        y_coords = [landmark.y * h for landmark in landmarks]
        bbox = [
            max(0, int(min(x_coords)) - 30),
            max(0, int(min(y_coords)) - 30),
            min(w, int(max(x_coords)) + 30),
            min(h, int(max(y_coords)) + 30)
        ]
        
        if self.model_type == 'ml':
            input_data = self._process_landmarks_for_ml(landmarks)
            prediction = self.model.predict(input_data)[0]
            prediction = int(prediction)
            confidence = 1.0  # ML models typically don't provide confidence
        else:
            input_tensor = self._process_image(frame, bbox)
            if input_tensor is None:
                return None, None, None, None
                
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                prediction = prediction.item()
                confidence = confidence.item()
        
        return self.labels_dict[prediction], bbox, landmarks, confidence

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['efficientnet', 'mobilenet', 'ml'],
                      help='Tipo de modelo a usar')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Ruta al archivo del modelo')
    args = parser.parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Load the appropriate class lookup based on model type
    if args.model_type == 'efficientnet':
        class_lookup_path = 'data/classs_lookup_v2.json' #original mapping
    else:
        class_lookup_path = 'data/class_lookup.json'
    
    # Load class lookup dictionary from JSON file
    with open(class_lookup_path, 'r') as f:
        class_lookup = {int(k): v.upper() for k, v in json.load(f).items()}

    # Initialize recognizer
    recognizer = HandGestureRecognizer(
        model_path=Path(args.model_path),
        model_type=args.model_type,
        class_lookup=class_lookup
    )
    
    print(f"Using device: {recognizer.device}")
    print(f"Model type: {args.model_type}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        prediction, bbox, landmarks, confidence = recognizer.predict(frame)
        
        if all(x is not None for x in [prediction, bbox, landmarks, confidence]):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            text = f"{prediction} ({confidence:.2f})"
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                       cv2.LINE_AA)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                   cv2.LINE_AA)
        
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#python src/interact_classifier.py --model_type ml --model_path results/randomforest_data1/evaluation_20241130_132749/checkpoints/random_forest_model.pkl
#python src/interact_classifier.py --model_type efficientnet --model_path results/efficientnet_v2_s_data3/evaluation_20241129_153938/checkpoints/Fine_Tuning_best_model.pth
#python src/interact_classifier.py --model_type efficientnet --model_path results/efficientnet_v2_s_data2/evaluation_20241129_222629/checkpoints/Fine_Tuning_best_model.pth