import os
import numpy as np
import pydicom
from scipy.io import loadmat
from tqdm import tqdm
import cv2

# Paths
BASE_DIR = r"/home/student/Desktop/SHAMA_Research/FUMPE"
CT_DIR = os.path.join(BASE_DIR, "CT_scans")
MASK_DIR = os.path.join(BASE_DIR, "GroundTruth")

# Output folder to save processed slices
OUTPUT_DIR = os.path.join(BASE_DIR, "Processed")
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

IMG_SIZE = 128  # Resize for faster training

data_info = []  # store info for later CSV if needed

for patient in tqdm(sorted(os.listdir(CT_DIR)), desc="Processing patients"):
    ct_path = os.path.join(CT_DIR, patient)
    mask_path = os.path.join(MASK_DIR, f"{patient}.mat")

    if not os.path.exists(mask_path):
        continue

    mat = loadmat(mask_path)
    mask_data = mat["Mask"]
    dcm_files = sorted([f for f in os.listdir(ct_path) if f.endswith(".dcm")])

    num_slices = min(len(dcm_files), mask_data.shape[-1])

    for idx in range(num_slices):
        dcm = pydicom.dcmread(os.path.join(ct_path, dcm_files[idx]))
        img = dcm.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize 0–1
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        mask = mask_data[..., idx]
        mask_binary = (mask > 0.001).astype(np.uint8)
        label = 1 if np.sum(mask_binary) > 0 else 0  # 1 = PE, 0 = normal

        # save numpy files (optional for training)
        np.save(os.path.join(OUTPUT_DIR, "images", f"{patient}_{idx:03d}.npy"), img)
        np.save(os.path.join(OUTPUT_DIR, "labels", f"{patient}_{idx:03d}.npy"), label)

        data_info.append((patient, idx, label))

print("Preprocessing complete.")
print(f"Total processed slices: {len(data_info)}")
print(f"Positive slices: {sum(l for _, _, l in data_info)}")
print(f"Negative slices: {len(data_info) - sum(l for _, _, l in data_info)}")