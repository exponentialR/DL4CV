import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
from PIL import Image
import cv2
import os

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
def load_model():
    global model  # Ensure model is accessible globally
    model_url_id = '1y1enpKZ6AcmP0bWYwsOLwzRIJM_rNts5'
    download_url = f'https://drive.google.com/uc?id={model_url_id}'
    output = 'resnet_c_d_p.pth'
    if not os.path.exists(output):
        
    # Download the model
        gdown.download(download_url, output, quiet=False)
    
    # Load the model architecture
    model = models.resnet18(weights=None)  # Explicitly set weights to None (avoids warnings)
    
    # Load checkpoint safely with `weights_only=True`
    checkpoint = torch.load(output, map_location=device, weights_only=True)

    # Dynamically adjust the FC layer to match the number of classes in the saved checkpoint
    num_features = model.fc.in_features
    num_classes = checkpoint["fc.bias"].shape[0]  # Get number of classes from checkpoint
    
    # Modify the last FC layer
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load the model state dictionary, allowing some mismatches
    model.load_state_dict(checkpoint, strict=False)
    
    # Move model to device
    model.to(device)
    model.eval()
    
# Function to process the video and save the output
def process_video(video_path, output_path):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    labels = ['Cat', 'Dog', 'Panda']

    # Load the video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process it
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Get the model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            label = labels[preds.item()]

        # Display the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object Detection - Animal Detection', frame)

        # Write the frame to the output video
        out.write(frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load the model
    load_model()
    video_id = '11iWDtEX8svf91K_1aRhUlGMojL_tFZ7q'

    download_url = f'https://drive.google.com/uc?id={video_id}'


    video_input = 'walking_dog.mov'

    gdown.download(download_url, video_input, quiet=False)
    # Path to the input video file

    # Path to the output video file
    output_path = 'processed_walking_dog.mp4'

    # Process the video and save the output
    process_video(video_input, output_path)
