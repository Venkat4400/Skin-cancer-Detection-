import os
import pandas as pd
import numpy as np
import kagglehub
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class SkinCancerDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.class_map = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
        }

    def __len__(self):
        return len(self.df)

    def apply_clahe(self, image):
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

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # Handle direct path or constructed path
            if 'image_dir' in row:
                img_path = os.path.join(row['image_dir'], row['image_id'] + '.jpg')
            else:
                 img_path = os.path.join(self.image_dir, row['image_id'] + '.jpg')
            
            image = Image.open(img_path).convert('RGB')
            
            # Apply CLAHE preprocessing
            image = self.apply_clahe(image)
            
        except (FileNotFoundError, OSError):
             # Fallback if image not found (shouldn't happen if setup is correct)
             print(f"Warning: Image not found/error at index {idx}")
             return torch.zeros(3, 224, 224), 0

        if self.transform:
            image = self.transform(image)
        
        label_str = row['dx']
        label = self.class_map[label_str]
        
        return image, label

def get_data_loaders(batch_size=32, data_dir=None):
    """
    Downloads dataset, pre-processes it, and returns DataLoaders.
    """
    if data_dir is None:
        # Check if local dataset exists first (common path)
        local_path = os.path.join(os.getcwd(), "skin-cancer-mnist-ham10000")
        if os.path.exists(local_path):
            data_dir = local_path
            print(f"Using local dataset at: {data_dir}")
        else:
            # Download dataset
            print("Downloading dataset...")
            try:
                path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
                print("Path to dataset files:", path)
                data_dir = path
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                # Fallback or exit
                raise e
    
    # Locate metadata and images
    metadata_path = None
    image_dirs = []
    
    for root, dirs, files in os.walk(data_dir):
        if 'HAM10000_metadata.csv' in files:
            metadata_path = os.path.join(root, 'HAM10000_metadata.csv')
        
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if len(jpg_files) > 0:
            image_dirs.append(root)

    if not metadata_path:
        raise FileNotFoundError("Could not find HAM10000_metadata.csv")
    
    print(f"Metadata found at: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Map image_id to directory
    image_path_map = {}
    for d in image_dirs:
        for f in os.listdir(d):
            if f.endswith('.jpg'):
                 image_id = os.path.splitext(f)[0]
                 image_path_map[image_id] = d
    
    def get_path(image_id):
        d = image_path_map.get(image_id)
        if d:
            return d
        return ""

    df['image_dir'] = df['image_id'].apply(get_path)
    
    # Filter out images not found
    df = df[df['image_dir'] != ""]
    print(f"Total images found: {len(df)}")

    # Stratified Split: 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Medical-Safe Augmentations (Updated to Resize 300)
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)), # Resize 300 as requested
        transforms.CenterCrop(224),    # Center crop 224
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((300, 300)), # Consistent Resize
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Datasets
    train_dataset = SkinCancerDataset(train_df, image_dir=None, transform=train_transform)
    val_dataset = SkinCancerDataset(val_df, image_dir=None, transform=val_test_transform)
    test_dataset = SkinCancerDataset(test_df, image_dir=None, transform=val_test_transform)

    # WeightedRandomSampler for imbalance in Training
    class_counts = train_df['dx'].value_counts().sort_index()
    # Inverse frequency weights
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    
    # Map weights to each sample in the dataset
    sample_weights = [class_weights[label] for label in train_df['dx']]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test logic
    try:
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
        print(f"Train batches: {len(train_loader)}")
        
        # Visualize one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}") # Should be [4, 3, 224, 224]
        
    except Exception as e:
        print(e)
