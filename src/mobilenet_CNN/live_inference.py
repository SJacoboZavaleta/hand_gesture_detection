# script used to live test classification of sign language gestures by model
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path


# Instantiate the model to be used for inference
model = models.mobilenet_v2(pretrained=False)

# If you made changes to the model, apply them (e.g., modifying the classifier)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 29)

# Load the saved model weights
model.load_state_dict(torch.load('./src/mobilenet_CNN/networks/mobilenet_sign_w_user_data_10_ep.pth'))

# Set the model to evaluation mode for inference
model.eval()

print("Model loaded successfully!")



# Load class lookup dictionary from JSON file
with open(Path(__file__).parent.parent.parent / 'data' / 'class_lookup.json', 'r') as f:
    class_lookup = json.load(f)


# Define the necessary transformations to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Convert the frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the necessary transformations
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    # Run the model on the input
    with torch.no_grad():  # Disable gradient computation since we're in inference mode
        output = model(input_tensor)
    
    # Get predicted class (index of the highest probability)
    _, predicted_class_idx = torch.max(output, 1)
    print(predicted_class_idx)
    # Get the class name
    predicted_class = class_lookup[str(predicted_class_idx.item())]
    
    # Display the result on the frame
    cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Webcam Feed", frame)
    
    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
