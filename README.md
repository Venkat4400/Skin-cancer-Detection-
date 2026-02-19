# DermaAI: Clinical-Grade Skin Lesion Analysis

DermaAI is a professional, hospital-grade AI tool designed for educational and screening purposes in dermatology. It leverages cutting-edge computer vision to analyze dermoscopic images and provide clinical risk assessments.

---

## 🛠 Technology Stack

### Backend (The Brain)
- **Framework**: [Flask](https://flask.palletsprojects.com/) (Python)
- **AI Engine**: [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/)
- **Model Architecture**: EfficientNet-V2-S (State-of-the-art accuracy)
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) for lesion localization
- **Calibration**: Temperature Scaling for reliable confidence scores
- **Image Processing**: OpenCV, PIL (Pillow), NumPy

### Frontend (The Interface)
- **Structure**: Semantic HTML5
- **Styling**: Vanilla CSS3 (Professional Medical UI/UX with Glassmorphism)
- **Interactivity**: Vanilla JavaScript (Async API calls, dynamic UI updates)
- **Typography**: Inter (Google Fonts)

---

## 🚀 Key Features

- **6-Stage Risk Assessment**: Categorizes lesions from Stage 0 (Safe/Benign) to Stage 5 (Critical/Urgent).
- **Grad-CAM Interpretability**: Generates a heatmap on the lesion to show exactly where the AI is looking.
- **Patient Demographics**: Integration of age, height, and weight for comprehensive reporting.
- **Premium Medical Reports**: Professional, animated reports generated instantly after analysis.
- **Responsive Design**: Fully optimized for Desktop and Mobile (including vertical column footer layout).
- **TTA (Test Time Augmentation)**: Runs multiple passes on flipped/rotated images for superior prediction stability.

---

## 📁 Project Structure

```text
Skin Cancer Detection/
├── app.py                # Main Flask Application
├── main.py               # Entry point for training/analysis
├── skin_cancer_model.pth # Pre-trained Model Weights
├── requirements.txt      # Python Dependencies
├── static/
│   ├── style.css         # Premium UI Styles
│   ├── script.js        # Frontend Logic
│   └── images/           # Asset Gallery (Disease codes, etc.)
├── templates/
│   └── index.html        # Main Application UI
└── src/                  # Core AI Engine
    ├── model.py          # Architecture Definition
    ├── gradcam.py        # Explainability Logic
    ├── calibration.py    # Confidence Calibration
    └── utils.py          # Helper functions
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- pip (Python Package Manager)

### 2. Clone the Project
```bash
git clone <repository-url>
cd "Skin Cancer Detection"
```

### 3. Setup Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
python app.py
```
*The app will be available at `http://127.0.0.1:5000`*

---

## 🩺 Usage Guide

1. **Input Vitals**: Enter patient age, height, and weight.
2. **Upload Image**: Select or drag a clear dermoscopic image of the lesion.
3. **Analyze**: Click "Initialize Analysis" to run the AI engine.
4. **Review Report**: Observe the final diagnosis, confidence score, and Grad-CAM heatmap.

> [!IMPORTANT]
> **Medical Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always evaluate with a certified dermatologist.
