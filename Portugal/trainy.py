import os, csv
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_H, IMG_W = 64, 160
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
SEQ_LEN = 8

CSV_PATH = "data/train_labels.csv"
DIR_TRAIN = "data/task_32/train"
DIR_IMAGES = "data/task_32/images"

BATCH_SIZE = 32
EPOCHS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_image(fname):
    # 1. try train dir with original name
    p1 = os.path.join(DIR_TRAIN, fname)
    if os.path.exists(p1):
        return p1

    # 2. try images dir with original name
    p2 = os.path.join(DIR_IMAGES, fname)
    if os.path.exists(p2):
        return p2

    # 3. CSV says train_000.png but disk has test_000.png
    if fname.startswith("train_"):
        alt = fname.replace("train_", "test_", 1)
        p3 = os.path.join(DIR_IMAGES, alt)
        if os.path.exists(p3):
            return p3
        p4 = os.path.join(DIR_TRAIN, alt)
        if os.path.exists(p4):
            return p4

    raise FileNotFoundError(f"no image for {fname}")

class CaptchaFixed(Dataset):
    def __init__(self, csv_path, transform=None):
        self.items = []
        self.transform = transform

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                label = row["answer"]
                self.items.append((fname, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        path = find_image(fname)
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)

        y = [0] * SEQ_LEN
        for i, ch in enumerate(label):
            y[i] = CHAR2IDX[ch]
        return img, torch.tensor(y, dtype=torch.long), os.path.basename(path)

transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

ds = CaptchaFixed(CSV_PATH, transform)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

class CNN8(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 16 * 40, SEQ_LEN * NUM_CLASSES)

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), SEQ_LEN, NUM_CLASSES)
        return x

model = CNN8().to(DEVICE)
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    total = 0.0
    for imgs, labels, _ in dl:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        out = model(imgs)          # B x 8 x C
        out = out.permute(0, 2, 1) # B x C x 8
        loss = criterion(out, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    if (epoch + 1) % 5 == 0:
        print(epoch + 1, total / len(dl))

torch.save({"model": model.state_dict()}, "cnn8.pt")
