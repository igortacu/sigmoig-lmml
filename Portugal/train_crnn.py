"""
Train CRNN model for CAPTCHA recognition
Uses CTC loss for sequence learning
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import csv
from collections import Counter

# Configuration
TRAIN_DIR = "data/synthetic_train"
TRAIN_CSV = os.path.join(TRAIN_DIR, "labels.csv")
IMG_HEIGHT = 64
IMG_WIDTH = 270
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character set
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 is blank for CTC
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank


class CaptchaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filename'], row['answer']))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        
        # Convert label to indices
        label_indices = [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]
        
        return img, torch.LongTensor(label_indices)


class CRNN(nn.Module):
    def __init__(self, img_height, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x32x135
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x16x67
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 256x8x67
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 512x4x67
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=512 * 4,  # 512 channels * 4 height
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # [batch, 512, 4, width]
        
        # Prepare for RNN: [width, batch, channels*height]
        batch, channel, height, width = conv.size()
        conv = conv.permute(3, 0, 1, 2)  # [width, batch, channel, height]
        conv = conv.reshape(width, batch, channel * height)
        
        # RNN
        output, _ = self.rnn(conv)  # [width, batch, hidden*2]
        
        # FC
        output = self.fc(output)  # [width, batch, num_classes]
        
        return output


def collate_fn(batch):
    """Custom collate function to handle variable length labels"""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    label_lengths = torch.LongTensor([len(label) for label in labels])
    labels = torch.cat(labels)
    
    return images, labels, label_lengths


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, labels, label_lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # [T, N, C]
        
        T, N, C = outputs.size()
        input_lengths = torch.full((N,), T, dtype=torch.long)
        
        # CTC loss
        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(outputs):
    """Decode CTC predictions to text"""
    _, preds = outputs.max(2)  # [T, N]
    preds = preds.transpose(1, 0)  # [N, T]
    
    texts = []
    for pred in preds:
        # Remove duplicates and blanks
        char_list = []
        prev = None
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev:  # 0 is blank
                if idx in IDX_TO_CHAR:
                    char_list.append(IDX_TO_CHAR[idx])
            prev = idx
        texts.append(''.join(char_list))
    
    return texts


def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found!")
        print("Run generate_dataset.py first to create synthetic training data")
        return
    
    print(f"Training on device: {DEVICE}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])
    
    # Dataset and dataloader
    dataset = CaptchaDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Training samples: {len(dataset)}")
    
    # Model
    model = CRNN(IMG_HEIGHT, NUM_CLASSES).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        loss = train_epoch(model, dataloader, criterion, optimizer, DEVICE)
        scheduler.step(loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss:.4f}")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'num_classes': NUM_CLASSES,
                'chars': CHARS
            }
            torch.save(checkpoint, 'crnn_best.pt')
            print(f"  -> Saved best model (loss: {loss:.4f})")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
