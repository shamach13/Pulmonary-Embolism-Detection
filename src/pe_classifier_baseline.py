import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

BASE_DIR = "/home/user30/SHAMA_Research/FUMPE/Processed"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

train_files = np.load("train_files.npy")
val_files = np.load("val_files.npy")

# -------------------------
# DATASET
# -------------------------
class PEDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)
        img = (img - 0.5) / 0.5
        label = np.load(os.path.join(LABEL_DIR, f))
        return img, torch.tensor(label, dtype=torch.float32)

train_loader = DataLoader(PEDataset(train_files), batch_size=16, shuffle=True)
val_loader = DataLoader(PEDataset(val_files), batch_size=16)

# -------------------------
# MODEL
# -------------------------
class PEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PEClassifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# TRAINING
# -------------------------
EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = torch.sigmoid(model(imgs).squeeze())
            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]

    print(
        f"Epoch {epoch+1} | "
        f"Recall: {recall_score(gts, preds_bin):.4f} | "
        f"F1: {f1_score(gts, preds_bin):.4f} | "
        f"AUC: {roc_auc_score(gts, preds):.4f}"
    )

torch.save(model.state_dict(), "pe_baseline.pth")
print("Baseline training complete.")
