import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by the gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def generate_heatmap(model, img_tensor, target_layer=None):
    """
    Helper function to generate and overlay heatmap.
    img_tensor: [1, 3, 224, 224]
    """
    # Create GradCAM object
    # For EfficientNet, target layer is usually the last convolutional layer
    # model.model.features[-1] is the last block
    if target_layer is None:
        target_layer = model.model.features[-1]
        
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(img_tensor)
    
    # Process for display
    # Resize heatmap to image size (224, 224)
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Convert heatmap to RGB coloring
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def overlay_heatmap(original_img_pil, heatmap_np, alpha=0.4):
    """
    original_img_pil: PIL Image (will be resized to 224x224)
    heatmap_np: numpy array from generate_heatmap
    """
    img = original_img_pil.resize((224, 224))
    img_np = np.array(img)
    
    # Convert RGB to BGR for OpenCV or keep RGB if handling manually. 
    # cv2.applyColorMap returns BGR.
    # Let's assume heatmap_np is BGR (from cv2).
    # Convert img_np to BGR
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap_np, alpha, 0)
    
    # Convert back to RGB for PIL/Display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)
    
