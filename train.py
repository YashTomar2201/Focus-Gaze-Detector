import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# --- CONFIGURATION ---
# If you unzipped data.zip directly in Colab, these paths are correct.
CSV_FILE = 'dataset_labels.csv'
IMG_DIR = 'CleanDataset'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10  # Increased slightly for better accuracy

# --- 1. The Model Architecture (Simple CNN) ---
# 

[Image of CNN architecture]

class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        # Input: 3 channels (RGB), Output: 2 values (Pitch, Yaw)
        
        # Layer 1: Convolution + ReLU + MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Convolution + ReLU + MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully Connected Layers
        # 64 channels * 16 * 16 (result of 64x64 image pooled twice)
        self.fc1 = nn.Linear(64 * 16 * 16, 128) 
        self.fc2 = nn.Linear(128, 2) # Final Output: Pitch and Yaw
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16) # Flatten the 2D images to 1D vector
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. The Data Loader ---
class EyesDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(), # Converts to 0-1 range
            # Optional: Add normalization for better stability
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # Fallback if an image is corrupt (rare but happens)
            return torch.zeros(3, 64, 64), torch.zeros(2)

        # Labels: Pitch (col 1), Yaw (col 2)
        y_label = torch.tensor(self.annotations.iloc[index, 1:].values.astype(float), dtype=torch.float32)
        image = self.transform(image)
        return image, y_label

# --- 3. Training Loop ---
def train():
    # AUTO-DETECT GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Check if data exists
    if not os.path.exists(CSV_FILE) or not os.path.exists(IMG_DIR):
        print("ERROR: Data not found! Did you run the unzip command?")
        return

    # Load Data
    dataset = EyesDataset(csv_file=CSV_FILE, root_dir=IMG_DIR)
    # num_workers=2 speeds up data loading in Colab
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize Model
    model = GazeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error (Standard for regression)

    print("Starting training...")
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train() # Set model to training mode
        
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # Clear old gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate Error
            loss.backward() # Backward pass (calculate gradients)
            optimizer.step() # Update weights
            
            running_loss += loss.item()
            
            # Print status every 100 batches
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 100:.5f}")
                running_loss = 0.0
        
        print(f"--- Epoch {epoch+1} Complete ---")

    # Save Model
    save_path = "my_gaze_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"SUCCESS: Model saved as {save_path}")
    print("Refresh the file browser on the left to see it!")

if __name__ == "__main__":
    train()