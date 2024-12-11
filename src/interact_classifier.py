import cv2
import mediapipe as mp
import pickle
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

class HandGestureRecognizer:
    def __init__(self, model_path, model_type, class_lookup):
        """
        Initialize the hand gesture recognizer.
        """
        self.model_type = model_type.lower()
        
        # MediaPipe hand configuration 
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector with more lenient settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            min_detection_confidence=0.3,
            max_num_hands=1
        )
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the appropriate model
        self.model = self._load_model(model_path)
        
        # Store model path for padding logic
        self.model_path = str(model_path)
        
        # Label mapping
        self.labels_dict = class_lookup
        
        # Setup image transforms
        if self.model_type == 'ml':
            self.transform = None
        else:
            image_size = (384, 384) if self.model_type == 'efficientnet' else (224, 224)
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])

    def _load_model(self, model_path):
        """Load and prepare model based on type."""
        try:
            if self.model_type == 'ml':
                with open(model_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    return loaded_data['model'] if isinstance(loaded_data, dict) else loaded_data
            
            # For PyTorch models
            if self.model_type == 'efficientnet':
                from efficientNet.model_class import ASLModel
                model = ASLModel(num_classes=29)
            elif self.model_type == 'mobilenet':
                from mobileNet.model_class import ASLModel
                model = ASLModel(num_classes=29)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
            model.eval().to(self.device)
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _process_landmarks(self, hand_landmarks, frame_width, frame_height):
        """Process hand landmarks for ML model."""
        # [Same implementation as before]
        data_aux = []
        x_ = []
        y_ = []
        
        # Collect coordinates
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)
        
        # Normalize coordinates
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))
        
        # Calculate bounding box
        x1 = int(min(x_) * frame_width) - 10
        y1 = int(min(y_) * frame_height) - 10
        x2 = int(max(x_) * frame_width) - 10
        y2 = int(max(y_) * frame_height) - 10
        
        return np.asarray(data_aux), (x1, y1, x2, y2)

    def _process_image_for_dl(self, frame, hand_landmarks):
        """Process image for deep learning models with dynamic padding."""
        if self.model_type == 'ml':
            return None
            
        try:
            orig_height, orig_width, _ = frame.shape

            # Collect hand landmark coordinates
            x_coords = [l.x * orig_width for l in hand_landmarks.landmark]
            y_coords = [l.y * orig_height for l in hand_landmarks.landmark]
            
            # Calculate center of the hand
            x_center = int((min(x_coords) + max(x_coords)) / 2)
            y_center = int((min(y_coords) + max(y_coords)) / 2)
            
            # Hand dimensions
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            
            # Check if path contains "data2" for padding strategy
            if "data2" in self.model_path:
                # Use 20% padding for data2 models
                box_size = int(max(hand_width, hand_height) * 1.2)
                box_size = max(box_size + (box_size % 2), 100)
            else:
                # For other models, use a larger crop (80-90% of frame)
                box_size = int(min(orig_width, orig_height) * 0.8)
            
            # Calculate square bbox coordinates
            half_size = box_size // 2
            x_min = max(0, x_center - half_size)
            x_max = min(orig_width, x_center + half_size)
            y_min = max(0, y_center - half_size)
            y_max = min(orig_height, y_center + half_size)
            
            # Adjust if box goes out of bounds
            if x_min == 0:
                x_max = min(orig_width, box_size)
            if x_max == orig_width:
                x_min = max(0, orig_width - box_size)
            if y_min == 0:
                y_max = min(orig_height, box_size)
            if y_max == orig_height:
                y_min = max(0, orig_height - box_size)
            
            # Crop hand region
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            if hand_img.size == 0:
                return None
                
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(hand_img)
            
            input_tensor = self.transform(pil_image)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Return tensor and bbox for potential visualization
            return input_tensor, (int(x_min), int(y_min), int(x_max), int(y_max))
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict(self, frame):
        """Predict hand gesture from frame."""
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None, None, None
        
        # Take first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Prepare landmarks for model
        if len(hand_landmarks.landmark) == 21:
            if self.model_type == 'ml':
                # For ML model
                input_data, bbox = self._process_landmarks(hand_landmarks, W, H)
                prediction = self.model.predict([input_data])[0]
                confidence = 1.0  # ML models typically don't provide confidence
            else:
                # For deep learning models
                image_data = self._process_image_for_dl(frame, hand_landmarks)
                
                if image_data is None:
                    return None, None, None, None
                
                input_tensor, bbox = image_data
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, prediction = torch.max(probabilities, 1)
                    prediction = prediction.item()
                    confidence = confidence.item()
            
            return self.labels_dict[int(prediction)], bbox, hand_landmarks, confidence
        
        return None, None, None, None

def get_model_config(option):
    # Definición de rutas predeterminadas y configuraciones de modelos
    model_configs = {
        1: {
            'model_type': 'ml',
            'model_path': 'results/randomforest_data1/evaluation_20241130_132749/checkpoints/random_forest_model.pkl'
        },
        2: {
            'model_type': 'efficientnet',
            'model_path': 'results/efficientnet_v2_s_data2/evaluation_20241206_174129/checkpoints/Fine_Tuning_best_model.pth'
        },
        3: {
            'model_type': 'efficientnet',
            'model_path': 'results/efficientnet_v2_s_data3/evaluation_20241206_152358/checkpoints/Fine_Tuning_best_model.pth'
        },
        4: {
            'model_type': 'mobilenet',
            'model_path': 'results/mobilenet_v2_data2/evaluation_20241207_144721/checkpoints/Fine_Tuning_best_model.pth'
        },
        5: {
            'model_type': 'mobilenet',
            'model_path': 'results/mobilenet_v2_data3/evaluation_20241207_112444/checkpoints/Fine_Tuning_best_model.pth'
        }
    }
    
    if option not in model_configs:
        raise ValueError(f"Opción de modelo inválida. Elija entre 1, 2, 3, 4 o 5.")
    
    return model_configs[option]

def main():
    # Define class lookup
    class_lookup = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 
        5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
        10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 
        15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 
        20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 
        25: "Z", 26: "DEL", 27: "NOTHING", 28: "SPACE"
    }

    parser = argparse.ArgumentParser(description='Hand Gesture Recognition')
    parser.add_argument('model_option', type=int, nargs='?', default=1, choices=[1, 2, 3, 4, 5],
                        help='Opción de modelo: 1-ML, 2-EfficientNet(data3), 3-EfficientNet(custom), 4-MobileNet(data2), 5-MobileNet(data3)')
    
    args = parser.parse_args()

    # Obtener la configuración del modelo basada en la opción
    model_config = get_model_config(args.model_option)

    # Inicializar reconocedor
    recognizer = HandGestureRecognizer(
        model_path=Path(model_config['model_path']),
        model_type=model_config['model_type'],
        class_lookup=class_lookup
    )
    
    print(f"Using device: {recognizer.device}")
    print(f"Model type: {model_config['model_type']}")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, bbox, hand_landmarks, confidence = recognizer.predict(frame)
        
        if result is not None and hand_landmarks is not None:
            # Draw hand landmarks and connections
            recognizer.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                recognizer.mp_hands.HAND_CONNECTIONS,
                recognizer.mp_drawing_styles.get_default_hand_landmarks_style(),
                recognizer.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw bounding box and prediction text
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 246, 235), 4)
            text = f"{result} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (252, 32, 225), 3,
                        cv2.LINE_AA)
        
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# Run
# python src/interact_classifier.py 1