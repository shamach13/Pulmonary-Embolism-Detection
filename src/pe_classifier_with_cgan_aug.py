# =========================================================
# STEP 3: PE Classifier with cGAN-Augmented Training Data
# =========================================================

import os
import numpy as np
import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "/home/user30/SHAMA_Research/FUMPE/Processed"

REAL_IMG_DIR = os.path.join(BASE_DIR, "images")
REAL_LABEL_DIR = os.path.join(BASE_DIR, "labels")

SYN_IMG_DIR = os.path.join(BASE_DIR, "synthetic_images")
SYN_LABEL_DIR = os.path.join(BASE_DIR, "synthetic_labels")

# Load original splits
train_files = list(np.load("train_files.npy"))
val_files = np.load("val_files.npy")

# Load synthetic files (TRAIN ONLY)
synthetic_files = sorted(os.listdir(SYN_IMG_DIR))

print("Real train samples:", len(train_files))
print("Synthetic train samples:", len(synthetic_files))

# -------------------------
# DATASET
# -------------------------
class PEDataset(Dataset):
    def __init__(self, real_files, synthetic_files=None, train=True):
        self.samples = []

        # Real data
        for f in real_files:
            self.samples.append((f, "real"))

        # Synthetic data (TRAIN ONLY)
        if train and synthetic_files is not None:
            for f in synthetic_files:
                self.samples.append((f, "synthetic"))

        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, source = self.samples[idx]

        if source == "real":
            img = np.load(os.path.join(REAL_IMG_DIR, fname)).astype(np.float32)
            label = np.load(os.path.join(REAL_LABEL_DIR, fname))
        else:
            img = np.load(os.path.join(SYN_IMG_DIR, fname)).astype(np.float32)
            label = np.load(os.path.join(SYN_LABEL_DIR, fname))

            # 🔑 FIX: resize synthetic images to 128×128
            if img.shape != (128, 128):
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)

        img = torch.tensor(img).unsqueeze(0)
        img = (img - 0.5) / 0.5
        return img, torch.tensor(label, dtype=torch.float32)

# Loaders
train_loader = DataLoader(
    PEDataset(train_files, synthetic_files, train=True),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    PEDataset(val_files, train=False),
    batch_size=16
)

# -------------------------
# MODEL (IDENTICAL TO BASELINE)
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

torch.save(model.state_dict(), "pe_classifier_cgan_aug.pth")
print("Training with cGAN augmentation complete.")
