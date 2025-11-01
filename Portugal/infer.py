import os, csv
from PIL import Image
import torch
import torchvision.transforms as T

IMG_H, IMG_W = 64, 160
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
SEQ_LEN = 8
TEST_DIR = "data/task_32/images"

class CNN8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(64, 128, 3, 1, 1), torch.nn.ReLU(), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(128, 256, 3, 1, 1), torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(256 * (IMG_H//4) * (IMG_W//4), SEQ_LEN * len(CHARS))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), SEQ_LEN, len(CHARS))
        return x

ckpt = torch.load("cnn8.pt", map_location="cpu")
model = CNN8()
model.load_state_dict(ckpt["model"])
model.eval()

transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

rows = [("filename", "answer")]

for fname in sorted(os.listdir(TEST_DIR)):
    if not fname.endswith(".png"):
        continue
    img = Image.open(os.path.join(TEST_DIR, fname)).convert("L")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)  # 1 x 8 x C
    idxs = out[0].argmax(1).tolist()
    text = "".join(CHARS[i] for i in idxs)
    rows.append((fname, text))

with open("submission.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("submission.csv written.")
