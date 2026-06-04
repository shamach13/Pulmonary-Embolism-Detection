# =========================================================
# Grad-CAM for PE Detection (Baseline vs cGAN-Augmented)
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "/home/user30/SHAMA_Research/FUMPE/Processed"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

OUT_DIR = "./gradcam_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

test_files = np.load("test_files.npy")

# -------------------------
# MODEL DEFINITION (EXACT MATCH)
# -------------------------
class PEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),   # net[0]
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1, 1),  # net[3]
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1), # net[6]  ← Grad-CAM layer
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# GRAD-CAM CLASS
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()
        output = self.model(x)
        output.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

# -------------------------
# LOAD MODEL
# -------------------------
def load_model(path):
    model = PEClassifier().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

baseline_model = load_model("pe_baseline.pth")
cgan_model = load_model("pe_classifier_cgan_aug.pth")

cam_baseline = GradCAM(baseline_model, baseline_model.net[6])
cam_cgan = GradCAM(cgan_model, cgan_model.net[6])

# -------------------------
# SELECT PE-POSITIVE TEST SLICES
# -------------------------
positive_files = [f for f in test_files if np.load(os.path.join(LABEL_DIR, f)) == 1]
samples = positive_files[:5]  # visualize 5 cases



# -------------------------
# GENERATE VISUALIZATIONS
# -------------------------
for idx, fname in enumerate(samples):

    img = np.load(os.path.join(IMG_DIR, fname)).astype(np.float32)

    img_norm = (img - 0.5) / 0.5

    img_tensor = torch.tensor(img_norm)\
        .unsqueeze(0)\
        .unsqueeze(0)\
        .to(DEVICE)

    img_tensor.requires_grad = True

    # Generate CAMs
    cam_b = cam_baseline.generate(img_tensor)
    cam_c = cam_cgan.generate(img_tensor)

    cam_b = cv2.resize(cam_b, (128, 128))
    cam_c = cv2.resize(cam_c, (128, 128))

    # Original image
    img_vis = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # Heatmaps
    heat_b = cv2.applyColorMap(
        np.uint8(255 * cam_b),
        cv2.COLORMAP_JET
    )

    heat_c = cv2.applyColorMap(
        np.uint8(255 * cam_c),
        cv2.COLORMAP_JET
    )

    # Overlay
    overlay_b = cv2.addWeighted(
        img_bgr, 0.6,
        heat_b, 0.4,
        0
    )

    overlay_c = cv2.addWeighted(
        img_bgr, 0.6,
        heat_c, 0.4,
        0
    )

    # Convert BGR → RGB
    overlay_b = cv2.cvtColor(overlay_b, cv2.COLOR_BGR2RGB)
    overlay_c = cv2.cvtColor(overlay_c, cv2.COLOR_BGR2RGB)

    # -------------------------
    # CREATE COMPARISON FIGURE
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    axes[0].imshow(img_vis, cmap='gray')
    axes[0].set_title("Original CTPA Slice")
    axes[0].axis('off')

    axes[1].imshow(overlay_b)
    axes[1].set_title("Baseline CNN")
    axes[1].axis('off')

    axes[2].imshow(overlay_c)
    axes[2].set_title("CNN + cGAN")
    axes[2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(
        OUT_DIR,
        f"gradcam_comparison_{idx+1}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

print("Grad-CAM comparison figures saved.")
