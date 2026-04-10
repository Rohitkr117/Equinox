import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tcn_model import TremorClassifierTCN

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models dir if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

class TremorMagnitudeDataset(Dataset):
    def __init__(self, csv_files, class_to_idx):
        self.csv_files = csv_files
        self.class_to_idx = class_to_idx
        
        self.data = []
        self.labels = []
        
        for file in self.csv_files:
            # Filename e.g., "Compound_Driving_slice1_2026.csv"
            basename = os.path.basename(file)
            label_name = basename.split('_slice')[0]
            
            if label_name not in self.class_to_idx:
                continue # Skip unknown labels if any (or we build dynamically, but here we expect it to match)
                
            try:
                df = pd.read_csv(file)
                if len(df) != 256:
                    print(f"Skipping {basename} - length is {len(df)}, not 256.")
                    continue
                    
                # Calculate Magnitudes
                # Accelerometer Magnitude: sqrt(ax^2 + ay^2 + az^2)
                accel_mag = np.sqrt(df['ax_g']**2 + df['ay_g']**2 + df['az_g']**2).values
                # Gyroscope Magnitude: sqrt(gx^2 + gy^2 + gz^2)
                gyro_mag = np.sqrt(df['gx_dps']**2 + df['gy_dps']**2 + df['gz_dps']**2).values
                
                # Stack into shape (2, 256)
                tensor_data = np.stack((accel_mag, gyro_mag), axis=0)
                
                self.data.append(tensor_data)
                self.labels.append(self.class_to_idx[label_name])
                
            except Exception as e:
                print(f"Error processing {basename}: {e}")
                
        self.data = torch.FloatTensor(np.array(self.data))
        self.labels = torch.LongTensor(np.array(self.labels))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train():
    print("--- Equinox Training Pipeline ---")
    csv_files = glob.glob(os.path.join(DATASET_DIR, '*.csv'))
    if not csv_files:
        print("No CSV files found in the dataset directory.")
        return

    # Automatically discover classes
    unique_classes = set()
    for f in csv_files:
        basename = os.path.basename(f)
        label_name = basename.split('_slice')[0]
        unique_classes.add(label_name)
        
    # Standardize class ordering (alphabetic)
    classes = sorted(list(unique_classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print(f"Found {len(csv_files)} dataset samples.")
    print(f"Detected classes: {classes}")
    
    # Save classes.json for inference UI
    classes_path = os.path.join(MODELS_DIR, 'classes.json')
    with open(classes_path, 'w') as f:
        json.dump(classes, f)
        
    # Build Dataset and DataLoader
    dataset = TremorMagnitudeDataset(csv_files, class_to_idx)
    if len(dataset) == 0:
        print("No valid 256-sample datasets found.")
        return
        
    # Batch size: if we have few samples, use a small batch
    batch_size = min(16, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset successfully created with {len(dataset)} valid samples.")
    
    # Initialize Model (input_channels=2 for Accel / Gyro magnitude)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = TremorClassifierTCN(input_channels=2, num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(dataloader)
        
        # Print every 10 epochs or first/last
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            
    print("\nTraining Complete!")
    
    # Save Model
    model_path = os.path.join(MODELS_DIR, 'tremor_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train()
