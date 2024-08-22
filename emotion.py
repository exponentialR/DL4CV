import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

# Define the emotion mapping
emotion_mapping = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

# Load the trained ResNet model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 7),
    nn.LogSoftmax(dim=1)
)

# Load the saved model weights
model.load_state_dict(torch.load('models/resnet_raf_face.pth', map_location=torch.device('cpu')))
model.eval()

# Transformation to be applied to each frame from the webcam
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply transformations
    input_tensor = transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        emotion = emotion_mapping[pred]

    # Display the result on the frame
    cv2.putText(frame, f'Emotion: {emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with the predicted emotion
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
