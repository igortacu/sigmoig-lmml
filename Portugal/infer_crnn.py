"""
Inference script for CAPTCHA recognition using CRNN model
"""
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import csv

# Configuration
TEST_DIR = "data/task_32/images"
MODEL_PATH = "crnn_best.pt"
OUTPUT_CSV = "submission.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CRNN(nn.Module):
    def __init__(self, img_height, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)
        
        # Prepare for RNN
        batch, channel, height, width = conv.size()
        conv = conv.permute(3, 0, 1, 2)
        conv = conv.reshape(width, batch, channel * height)
        
        # RNN
        output, _ = self.rnn(conv)
        
        # FC
        output = self.fc(output)
        
        return output


def decode_prediction(output, idx_to_char):
    """Decode CTC prediction to text"""
    _, pred = output.max(2)  # [T, N]
    pred = pred.squeeze(1)  # [T]
    
    # Remove duplicates and blanks
    char_list = []
    prev = None
    for idx in pred:
        idx = idx.item()
        if idx != 0 and idx != prev:  # 0 is blank
            if idx in idx_to_char:
                char_list.append(idx_to_char[idx])
        prev = idx
    
    return ''.join(char_list)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Train the model first using train_crnn.py")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Get model parameters
    img_height = checkpoint['img_height']
    img_width = checkpoint['img_width']
    num_classes = checkpoint['num_classes']
    chars = checkpoint['chars']
    
    # Create character mapping
    idx_to_char = {idx + 1: char for idx, char in enumerate(chars)}
    
    # Load model
    model = CRNN(img_height, num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Image size: {img_height}x{img_width}")
    print(f"Character set: {chars}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    
    # Process test images
    results = []
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])
    
    print(f"\nProcessing {len(test_files)} test images...")
    
    for i, filename in enumerate(test_files):
        img_path = os.path.join(TEST_DIR, filename)
        
        # Load and preprocess image
        img = Image.open(img_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
        
        # Decode prediction
        text = decode_prediction(output, idx_to_char)
        results.append((filename, text))
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(test_files)} images")
    
    # Save results
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'answer'])
        writer.writerows(results)
    
    print(f"\nDone! Results saved to {OUTPUT_CSV}")
    print(f"Total predictions: {len(results)}")
    
    # Show first few predictions
    print("\nFirst 10 predictions:")
    for filename, text in results[:10]:
        print(f"  {filename}: {text}")


if __name__ == "__main__":
    main()
