import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import sys
import os

def predict(image_path, model_path='best_flower_model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    classes = checkpoint['classes']

    # Initialize model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted[0]].item()

    print(f"Prediction: {classes[predicted[0]]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")
