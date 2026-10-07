import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import UncertaintyConfig
from .model import UncertaintyUNet
from .dataset import RestorationErrorDataset
from .losses import UncertaintyLoss

def train(data_dir, config_path=None):
    config = UncertaintyConfig(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Dataset
    batch_size = config.get('training', 'batch_size')
    img_size = config.get('dataset', 'image_size')
    
    train_dataset = RestorationErrorDataset(data_dir, image_size=img_size, is_train=True)
    val_dataset = RestorationErrorDataset(data_dir, image_size=img_size, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = UncertaintyUNet(
        in_channels=config.get('model', 'in_channels'),
        out_channels=config.get('model', 'out_channels'),
        base_filters=config.get('model', 'base_filters')
    ).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('training', 'learning_rate')))
    criterion = UncertaintyLoss(loss_type='smooth_l1').to(device)
    
    # Training Loop
    epochs = config.get('training', 'epochs')
    save_dir = Path(config.get('training', 'save_dir'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = config.get('training', 'patience')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_uncertainty_model.pth")
            print("Saved new best model.")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Uncertainty Estimator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to research_restored_output")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    
    args = parser.parse_args()
    
    # Need to run from project root
    os.chdir(Path(__file__).parent.parent.parent)
    train(args.data_dir, args.config)
