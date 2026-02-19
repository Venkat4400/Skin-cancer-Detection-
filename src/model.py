import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        """
        Initializes the EfficientNet-V2-S model (High Accuracy & Efficiency).
        Args:
            num_classes (int): Number of output classes (default: 7).
            pretrained (bool): Whether to use pretrained weights (default: True).
        """
        super(SkinCancerModel, self).__init__()
        
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)
        
        # Modify the classifier head
        # EfficientNet-V2-S classifier input features: 1280
        in_features = self.model.classifier[1].in_features
        
        # Custom Head for better regularization and feature extraction
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test
    model = SkinCancerModel(num_classes=7)
    print(model)
    # EfficientNet-V2-S typically uses 384x384 for max acc, but 224x224 is standard for speed/transfer
    dummy_input = torch.randn(1, 3, 224, 224) 
    output = model(dummy_input)
    print("Output shape:", output.shape)
