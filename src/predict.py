import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import argparse
from src.model import SkinCancerModel
from src.utils import get_label_mapping

def apply_clahe(image):
    # Convert PIL to OpenCV (RGB -> LAB)
    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final)

def predict_image(model_path, image_path, device='cpu'):
    # Load Model
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    model = SkinCancerModel(num_classes=len(classes), pretrained=False)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Make sure the model architecture matches the checkpoint.")
        return

    model.to(device)
    model.eval()

    # Process Image
    try:
        image = Image.open(image_path).convert('RGB')
        image = apply_clahe(image)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    # Transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    inputs = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    predicted_class = classes[idx.item()]
    confidence_score = conf.item() * 100
    
    mapping = get_label_mapping().get(predicted_class, {})
    diagnosis = mapping.get('diagnosis', predicted_class)

    print(f"\n--- Prediction Result ---")
    print(f"Image: {image_path}")
    print(f"Algorithm: EfficientNet-V2-S")
    print(f"Class: {predicted_class} ({diagnosis})")
    print(f"Confidence: {confidence_score:.2f}%")
    
    if confidence_score < 45:
        print("WARNING: Low confidence. Result may be unreliable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='skin_cancer_model.pth', help='Path to model checkpoint')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_image(args.model, args.image, device)
