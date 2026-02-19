import argparse
import torch
import os
from src.data_loader import get_data_loaders
from src.model import SkinCancerModel
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image

def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Detection System")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--image', type=str, help='Path to image for prediction (required for predict mode)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training (Default: 30)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (Lowered for B4 memory)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default='skin_cancer_model.pth', help='Path to save/load model')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    if args.mode == 'train':
        print(f"Initializing Training Pipeline on {device}...")
        print("Note: EfficientNet-B4 requires more VRAM. If OOM occurs, reduce batch_size.")
        
        # Data Loaders
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
        
        # Model
        model = SkinCancerModel(num_classes=len(classes))
        
        # Train
        model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            num_epochs=args.epochs, 
            learning_rate=args.lr,
            save_path=args.model_path,
            patience=7 # Early stopping patience
        )
        
        # Final Evaluation on Test Set
        print("\nRunning Final Evaluation on Test Set...")
        evaluate_model(model, test_loader, device, classes)
        
        print("Training Complete! You can now run the app.")
        
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image argument is required for prediction mode.")
            return

        print(f"Loading model from {args.model_path}...")
        if not os.path.exists(args.model_path):
             print("Error: Model file not found. Please train the model first.")
             return

        # Pretrained=False because we are loading custom weights
        model = SkinCancerModel(num_classes=len(classes), pretrained=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        predict_image(model, args.image, device, classes)

if __name__ == "__main__":
    main()
