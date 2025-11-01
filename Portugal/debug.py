import os
from PIL import Image
import torch
import torchvision.transforms as T

CKPT_PATH = "crnn_captcha.pt"
IMG_PATH = "data/train/train_000.png"   # update if name differs

# same charset
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

class CRNN(torch.nn.Module):
    def __init__(self, img_h, n_channels, n_classes):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, 64, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.ReLU(True),
        )
        self.rnn = torch.nn.LSTM(
            input_size=256 * (64 // 4),
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = torch.nn.Linear(256 * 2, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2)
        x = x.reshape(w, b, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = CRNN(ckpt["img_h"], 1, len(ckpt["chars"]) + 1)
model.load_state_dict(ckpt["model"])
model.eval()

transform = T.Compose([
    T.Resize((ckpt["img_h"], ckpt["img_w"])),
    T.ToTensor(),
])

img = Image.open(IMG_PATH).convert("L")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(img)  # T x 1 x C
probs = logits.softmax(2).squeeze(1)  # T x C
top = probs.argmax(1).tolist()

print("time steps:", len(top))
print("top idx:", top[:30])
