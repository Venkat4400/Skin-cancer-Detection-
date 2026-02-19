import os
import time
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import cv2
from torchvision import transforms
from src.model import SkinCancerModel
from src.calibration import ModelWithTemperature
from src.gradcam import generate_heatmap, overlay_heatmap
from src.utils import get_label_mapping

# Standard Flask Structure (defaults to templates/ and static/)
app = Flask(__name__)
# 1. Fix CORS properly
CORS(app, resources={r"/*": {"origins": "*"}})

# Global Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MODEL_PATH = 'skin_cancer_model.pth' # Updated model path

# --- Preprocessing & TTA ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Base Transform (Resize 300 + CenterCrop 224 to match training)
base_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def apply_clahe(image):
    try:
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
    except Exception as e:
        print(f"CLAHE Warning: {e}")
        return image # Fallback to original

def get_clinical_info(diagnosis_key, confidence):
    mapping = get_label_mapping()
    info = mapping.get(diagnosis_key, {})
    
    diagnosis = info.get('diagnosis', 'Unknown')
    diag_type = info.get('type', 'Unknown')
    
    # Defaults
    stage = 1
    stage_label = "Stage 1 - Benign"
    risk_level = "Safe"
    color_code = "#10b981" # Green
    summary = "The lesion appears benign with no alarming features."
    recommendation = "Routine annual skin check-up recommended."

    # --- 6-Stage Premium Severity Logic ---
    
    # Benign Classes: 'nv', 'bkl', 'df', 'vasc'
    if diagnosis_key in ['nv', 'bkl', 'df', 'vasc']:
        if confidence >= 90:
            stage = 0
            stage_label = "Stage 0"
            risk_level = "Safe"
            color_code = "#10b981" # Emerald Green (Safe)
            summary = f"Diagnosis: {diagnosis}. High confidence ({confidence:.1f}%) indicates a safe, benign lesion."
            recommendation = "No medical action required. Routine self-exams."
        elif confidence >= 75:
            stage = 1
            stage_label = "Stage 1"
            risk_level = "Benign"
            color_code = "#84cc16" # Lime Green (Benign)
            summary = f"Likely benign {diagnosis} ({confidence:.1f}%). Typical presentation for this non-cancerous lesion."
            recommendation = "Routine observation. Monitor for any changes."
        else:
            stage = 2
            stage_label = "Stage 2"
            risk_level = "Low Risk"
            color_code = "#eab308" # Yellow (Low Risk)
            summary = f"Probable {diagnosis}, but confidence ({confidence:.1f}%) warrants a closer look."
            recommendation = "Consider a non-urgent check-up if lesion changes."

    # Pre-Cancerous: 'akiec'
    elif diagnosis_key == 'akiec':
        stage = 3
        stage_label = "Stage 3"
        risk_level = "Moderate Risk"
        color_code = "#f97316" # Orange (Moderate Risk)
        summary = "Actinic Keratosis detected. This is a pre-cancerous skin lesion."
        recommendation = "Dermatologist visit recommended for potential cryotherapy treatment."

    # Malignant: 'bcc'
    elif diagnosis_key == 'bcc':
        if confidence >= 80:
            stage = 4
            stage_label = "Stage 4"
            risk_level = "High Risk"
            color_code = "#ef4444" # Red (High Risk)
            summary = f"Strong indicator of Basal Cell Carcinoma ({confidence:.1f}%). Common but requires treatment."
            recommendation = "Schedule a biopsy with a dermatologist soon."
        else:
            stage = 3
            stage_label = "Stage 3"
            risk_level = "Moderate Risk"
            color_code = "#f97316" # Orange (Moderate Risk)
            summary = "Features consistent with Basal Cell Carcinoma, though not definitive."
            recommendation = "Medical consultation advised to rule out malignancy."

    # Malignant (Aggressive): 'mel'
    elif diagnosis_key == 'mel':
        if confidence >= 80:
            stage = 5
            stage_label = "Stage 5"
            risk_level = "Critical"
            color_code = "#b91c1c" # Deep Red (Critical)
            summary = f"CRITICAL: High probability of Melanoma ({confidence:.1f}%). Most serious skin cancer."
            recommendation = "IMMEDIATE medical attention required. Do not delay."
        else:
            stage = 4
            stage_label = "Stage 4"
            risk_level = "High Risk"
            color_code = "#ef4444" # Red (High Risk)
            summary = "Suspicious lesion with features suggestive of Melanoma."
            recommendation = "Urgent dermatologist appointment required."

    return {
        "diagnosis": diagnosis,
        "stage_id": stage, # Numeric ID for frontend logic
        "stage": f"Stage {stage}",
        "stage_description": stage_label,
        "risk_level": risk_level,
        "color_code": color_code,
        "summary": summary,
        "recommendation": recommendation
    }



# --- Load Model Once ---
print(f"Initializing Model on {DEVICE}...")
base_model = SkinCancerModel(num_classes=len(CLASSES), pretrained=False)
model = None

# Try loading calibrated first, then standard
if os.path.exists('skin_cancer_model_calibrated.pth'):
    try:
        print("Loading Calibrated Model...")
        model = ModelWithTemperature(base_model)
        model.load_state_dict(torch.load('skin_cancer_model_calibrated.pth', map_location=DEVICE))
    except Exception as e:
        print(f"Calibration Load Failed: {e}")

if model is None and os.path.exists(MODEL_PATH):
    try:
        print("Loading Standard Model...")
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = base_model
    except Exception as e:
        print(f"Standard Load Failed: {e}")

if model:
    model.to(DEVICE)
    model.eval()
    print("Model Loaded Successfully.")
else:
    print("CRITICAL WARNING: No model loaded. API will fail.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 3. Proper Error Handling (Global Try/Except)
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not valid. Please contact administrator."
            }), 500

        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        # 4. Relaxed Image Validation
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception:
             return jsonify({"success": False, "error": "Invalid image format. Use JPG/PNG."}), 400
        
        # Preprocessing
        image_clahe = apply_clahe(image)
        
        # --- TTA Loop (Robust) ---
        imgs = [
            image_clahe,
            image_clahe.transpose(Image.FLIP_LEFT_RIGHT),
            image_clahe.rotate(90),
            image_clahe.rotate(-90),
            image_clahe.transpose(Image.FLIP_TOP_BOTTOM)
        ]
        
        tensors = [base_transform(img).to(DEVICE) for img in imgs]
        batch = torch.stack(tensors) 
        
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            avg_probs = torch.mean(probs, dim=0) 
            
        top_prob, top_idx = torch.max(avg_probs, 0)
        best_conf = top_prob.item() * 100
        best_class = CLASSES[top_idx.item()]
        
        # Robust Grad-CAM
        heatmap_base64 = None
        try:
            input_tensor = tensors[0].unsqueeze(0).clone().detach().requires_grad_(True)
            heatmap = generate_heatmap(model, input_tensor)
            overlay_img = overlay_heatmap(image_clahe, heatmap)
            buffered = io.BytesIO()
            overlay_img.save(buffered, format="JPEG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"GradCAM Skipped: {e}")

        # Build Response
        clinical_info = get_clinical_info(best_class, best_conf)
        
        # 2. Consistent JSON Structure (Strict 5-Stage Format)
        response = {
            "success": True,
            "diagnosis": clinical_info['diagnosis'],
            "confidence": round(best_conf, 2),
            "risk_level": clinical_info['risk_level'],
            "stage_id": clinical_info['stage_id'], # Critical for frontend theming
            "stage": clinical_info['stage'],
            "color_code": clinical_info['color_code'],
            "summary": clinical_info['summary'],
            "recommendation": clinical_info['recommendation'],
            "heatmap_image": heatmap_base64
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"CRASH PREVENTION: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal Server Error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
