import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
import os
import copy
import numpy as np
import torch.nn.functional as F
from src.utils import plot_training_history
from src.calibration import ModelWithTemperature

class WeightedFocalLoss(nn.Module):
    "Non-weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha[targets] * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Simple Focal Loss for Multi-Class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        return F_loss

def train_model(model, train_loader, val_loader, device, num_epochs=30, learning_rate=3e-4, patience=5, save_path="best_model.pth"):
    
    # 1. Loss Function (Focal Loss to handle hard examples)
    # Note: Class imbalance is already handled by WeightedRandomSampler in DataLoader
    criterion = FocalLoss(gamma=2.0)
    
    # 2. Optimizer (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 3. Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 4. Mixed Precision Scaler
    scaler = amp.GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    model = model.to(device)
    
    print(f"Starting training on {device}...")
    
    # --- STAGE 1: FREEZE BODY, TRAIN HEAD (3 Epochs) ---
    print("Stage 1: Training Head Only (3 Epochs)...")
    for param in model.model.features.parameters():
        param.requires_grad = False
        
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        loop = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/3")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            loop.set_postfix(loss=loss.item())
            
    print("Stage 1 Complete. Unfreezing all layers...")
    
    # --- STAGE 2: FINE-TUNE ALL LAYERS ---
    for param in model.parameters():
        param.requires_grad = True
        
    # Reset Optimizer/Scheduler for full training? 
    # Usually keep them, but maybe lower LR. 
    # For simplicity, we continue with current state but unfreeze.
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            loop = tqdm(dataloader, desc=f"{phase} Phase")
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                loop.set_postfix(loss=loss.item())
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"Saving best model with Acc: {best_acc:.4f}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
            
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Reload best model
    model.load_state_dict(best_model_wts)
    
    # --- POST-TRAINING: TEMPERATURE SCALING ---
    print("Performing Temperature Scaling...")
    calibrated_model = ModelWithTemperature(model)
    calibrated_model.set_temperature(val_loader, device)
    
    # Save Calibrated Model State
    torch.save(calibrated_model.state_dict(), "skin_cancer_model_calibrated.pth")
    print("Saved calibrated model to skin_cancer_model_calibrated.pth")
    
    plot_training_history(history, save_path="training_history.png")
    
    return calibrated_model, history
