# =========================================================
# STEP 4: Final Test Set Evaluation (Baseline vs cGAN-Aug)
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "/home/user30/SHAMA_Research/FUMPE/Processed"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

test_files = np.load("test_files.npy")

# -------------------------
# DATASET
# -------------------------
class PEDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = np.load(os.path.join(IMG_DIR, f)).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)
        img = (img - 0.5) / 0.5
        label = np.load(os.path.join(LABEL_DIR, f))
        return img, torch.tensor(label, dtype=torch.float32)

test_loader = DataLoader(PEDataset(test_files), batch_size=16)

# -------------------------
# MODEL DEFINITION (same as training)
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

# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(model_path, name):
    model = PEClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    preds, gts = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = torch.sigmoid(model(imgs).squeeze())
            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]

    print(f"\n{name} RESULTS (TEST SET)")
    print("Accuracy:", accuracy_score(gts, preds_bin))
    print("Recall:  ", recall_score(gts, preds_bin))
    print("F1:      ", f1_score(gts, preds_bin))
    print("AUC:     ", roc_auc_score(gts, preds))

    return gts, preds, preds_bin
# -------------------------
# RUN EVALUATION
# -------------------------
# Evaluate both models
gts_base, preds_base, preds_bin_base = evaluate("pe_baseline.pth", "BASELINE")

gts_aug, preds_aug, preds_bin_aug = evaluate("pe_classifier_cgan_aug.pth", "cGAN-AUGMENTED")

# =========================================================
# ROC CURVE PLOTTING
# =========================================================

fpr_base, tpr_base, _ = roc_curve(gts_base, preds_base)
fpr_aug, tpr_aug, _ = roc_curve(gts_aug, preds_aug)

auc_base = roc_auc_score(gts_base, preds_base)
auc_aug = roc_auc_score(gts_aug, preds_aug)

plt.figure(figsize=(8,6))

plt.plot(
    fpr_base,
    tpr_base,
    linewidth=2,
    label=f'Baseline CNN (AUC = {auc_base:.3f})'
)

plt.plot(
    fpr_aug,
    tpr_aug,
    linewidth=2,
    label=f'CNN + cGAN Augmentation (AUC = {auc_aug:.3f})'
)

plt.plot([0,1], [0,1], linestyle='--')

plt.style.use('default')
plt.grid(alpha=0.3)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Pulmonary Embolism Detection")
plt.legend(loc="lower right")

plt.savefig("final_roc_curve.png", dpi=300)

print("\nROC curve saved as final_roc_curve.png")


# =========================================================
# CONFUSION MATRIX PLOTTING
# =========================================================

from sklearn.metrics import confusion_matrix

cm_base = confusion_matrix(gts_base, preds_bin_base)
cm_aug = confusion_matrix(gts_aug, preds_bin_aug)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# -------------------------
# BASELINE CONFUSION MATRIX
# -------------------------
im1 = axes[0].imshow(cm_base, cmap='Blues')

axes[0].set_title("Baseline CNN")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

axes[0].set_xticks([0,1])
axes[0].set_yticks([0,1])

axes[0].set_xticklabels(["Non-PE", "PE"])
axes[0].set_yticklabels(["Non-PE", "PE"])

for i in range(2):
    for j in range(2):
        axes[0].text(
            j,
            i,
            cm_base[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

# -------------------------
# AUGMENTED CONFUSION MATRIX
# -------------------------
im2 = axes[1].imshow(cm_aug, cmap='Greens')

axes[1].set_title("CNN + cGAN Augmentation")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

axes[1].set_xticks([0,1])
axes[1].set_yticks([0,1])

axes[1].set_xticklabels(["Non-PE", "PE"])
axes[1].set_yticklabels(["Non-PE", "PE"])

for i in range(2):
    for j in range(2):
        axes[1].text(
            j,
            i,
            cm_aug[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

print("Confusion matrix saved as confusion_matrix.png")

