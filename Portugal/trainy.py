import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "data/seed_chars"
assert os.path.exists(DATA_ROOT), "run build_char_from_seed.py first"

BATCH = 32
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])

ds = datasets.ImageFolder(DATA_ROOT, transform=tf)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

class CharNet(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 16 * 16, n_cls)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CharNet(len(ds.classes)).to(DEVICE)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    loss_sum = 0.0
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        out = model(xb)
        loss = crit(out, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f"{epoch+1:02d} {loss_sum/len(dl):.4f}")

torch.save({"model": model.state_dict(), "classes": ds.classes}, "char_from_seed.pt")
print("[ok] saved char_from_seed.pt")
