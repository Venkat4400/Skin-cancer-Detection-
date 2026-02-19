import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def plot_training_history(history, save_path=None):
    """Plots training and validation accuracy and loss."""
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'go-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'go-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Computes and plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_label_mapping():
    """Returns the mapping of classes to Benign/Malignant with Clinical Reasoning."""
    
    mapping = {
        'akiec': {
            'diagnosis': 'Actinic Keratosis',
            'type': 'Pre-Cancerous / Malignant Potential',
            'reasoning': 'Rough, scaly patch on skin caused by chronic sun exposure. While initially benign, it has the potential to transform into Squamous Cell Carcinoma. Texture patterns indicate keratin buildup.'
        },
        'bcc': {
            'diagnosis': 'Basal Cell Carcinoma',
            'type': 'Malignant',
            'reasoning': 'Presence of pearly nodules, telangiectasia (visible blood vessels), and potentially ulceration. Borders are often rolled. This is the most common form of skin cancer but rarely metastasizes.'
        },
        'bkl': {
            'diagnosis': 'Benign Keratosis',
            'type': 'Benign',
            'reasoning': 'Waxy, stuck-on appearance often associated with Seborrheic Keratosis. Sharp borders and uniform pigmentation or "milky cysts" observed. No signs of invasive growth.'
        },
        'df': {
            'diagnosis': 'Dermatofibroma',
            'type': 'Benign',
            'reasoning': 'Firm, solitary nodule that often dimples when pinched (dimple sign). Characterized by a central white scar-like patch and a peripheral pigment network.'
        },
        'mel': {
            'diagnosis': 'Melanoma',
            'type': 'Malignant (High Risk)',
            'reasoning': 'High asymmetry, irregular borders, and color variegation (multiple colors including black, blue, or white). Pigment network is atypical/broad. Suggests invasive melanocytic activity.'
        },
        'nv': {
            'diagnosis': 'Melanocytic Nevus',
            'type': 'Benign',
            'reasoning': 'Symmetrical shape, regular borders, and uniform coloration. Typical pigment network or homogenous pattern. Common mole structure without signs of dysplasia.'
        },
        'vasc': {
            'diagnosis': 'Vascular Lesion',
            'type': 'Benign',
            'reasoning': 'Red to purple lacunae separated by fibrous septa. Absence of pigment network. Appearance inconsistent with melanocytic tumors, likely a hemangioma or similar vascular anomaly.'
        }
    }
    return mapping
