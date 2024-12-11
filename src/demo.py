import gradio as gr
import cv2
import torch
import numpy as np
from interact_classifier import HandGestureRecognizer, get_model_config
from pathlib import Path

# Define class lookup (matching the one in interact_classifier.py)
CLASS_LOOKUP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 
    25: "Z", 26: "DEL", 27: "NOTHING", 28: "SPACE"
}

class GestureRecognitionApp:
    def __init__(self):
        self.recognizer = None
        self.current_model_option = None
        
    def initialize_model(self, model_option):
        """Initialize the model based on selected option"""
        try:
            # Get model configuration using the get_model_config function
            model_config = get_model_config(model_option)
            
            # Create HandGestureRecognizer instance
            self.recognizer = HandGestureRecognizer(
                model_path=Path(model_config['model_path']),
                model_type=model_config['model_type'],
                class_lookup=CLASS_LOOKUP
            )
            
            self.current_model_option = model_option
            return f"Model {model_config['model_type']} initialized successfully"
        except Exception as e:
            return f"Error initializing model: {str(e)}"
    
    def recognize_gesture(self, model_option, image):
        """Perform gesture recognition on an input image"""
        try:
            # Initialize model if not already done or model changed
            if (self.recognizer is None or 
                self.current_model_option != model_option):
                init_result = self.initialize_model(model_option)
                if "Error" in init_result:
                    return None, init_result, None
            
            # Convert image to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform prediction
            result, bbox, hand_landmarks, confidence = self.recognizer.predict(frame)
            
            # If no prediction, return appropriate message
            if result is None:
                return image, "No hand detected", None
            
            # Draw results on frame
            if bbox is not None and hand_landmarks is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 246, 235), 4)
                text = f"{result} ({confidence:.2f})"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (252, 32, 225), 3,
                            cv2.LINE_AA)
            
            # Convert back to RGB for Gradio
            result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return result_image, f"Predicted: {result} (Confidence: {confidence:.2f})", {
                "Prediction": result,
                "Confidence": round(confidence, 2)
            }
        
        except Exception as e:
            return None, f"Error: {str(e)}", None

# Create app instance
app = GestureRecognitionApp()

# Gradio Interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Hand Gesture Recognition")
        
        with gr.Row():
            # Model selection dropdown
            model_dropdown = gr.Dropdown(
                [1, 2, 3, 4, 5], 
                label="Select Model", 
                value=1,
                info="1-ML, 2-EfficientNet(data2), 3-EfficientNet(data3), 4-MobileNet(data2), 5-MobileNet(data3)"
            )
            
            # Image input
            image_input = gr.Image(
                sources=["upload"],#"webcam" 
                type="pil", 
                label="Hand Gesture Image"
            )
        
        # Outputs
        with gr.Row():
            output_image = gr.Image(label="Processed Image")
            prediction_output = gr.Textbox(label="Prediction Result")
        
        # Prediction details
        prediction_details = gr.JSON(label="Prediction Details")
        
        # Recognize button
        recognize_btn = gr.Button("Recognize Gesture")
        
        # Event handler
        recognize_btn.click(
            fn=app.recognize_gesture, 
            inputs=[model_dropdown, image_input], 
            outputs=[output_image, prediction_output, prediction_details]
        )
    
    return demo

# Launch the interface
demo = create_interface()
demo.launch(
    share=True,
    server_port=7860,
    show_error=True,
    debug=True
)

# Go to http://127.0.0.1:7860
# Online  https://50096faaa85185bd62.gradio.live