import os
import numpy as np
from scipy.io import loadmat
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt

# Base directories
BASE_DIR = r"/home/student/Desktop/SHAMA_Research/FUMPE"
CT_DIR = os.path.join(BASE_DIR, "CT_scans")
GT_DIR = os.path.join(BASE_DIR, "GroundTruth")

count_pos, count_neg, missing = 0, 0, 0
patients = sorted([p for p in os.listdir(CT_DIR) if os.path.isdir(os.path.join(CT_DIR, p))])

for patient in tqdm(patients, desc="Checking patients"):
    ct_path = os.path.join(CT_DIR, patient)
    gt_path = os.path.join(GT_DIR, f"{patient}.mat")

    if not os.path.exists(gt_path):
        print(f"⚠️ Missing GroundTruth for {patient}")
        missing += 1
        continue

    # Load ground truth mask (.mat file)
    try:
        mat = loadmat(gt_path)
        # find key (usually 'GT' or 'mask')
        key = [k for k in mat.keys() if not k.startswith('__')][0]
        mask_data = mat[key]  # shape: (rows, cols, slices)
    except Exception as e:
        print(f"Error reading {gt_path}: {e}")
        continue

    # Count PE-positive and PE-negative slices
    num_slices = mask_data.shape[-1]
    result ={}
    local_pos = 0
    local_neg = 0
    for i in range(num_slices):
        mask = mask_data[..., i]
        if np.sum(mask) > 0:
            local_pos+=1
            count_pos += 1
        else:
       	    local_neg +=1
            count_neg += 1
    result[patient]=[local_pos, local_neg]
total = count_pos + count_neg
print(result)

print("\n=== Dataset Summary ===")
print(f"Total slices processed: {total}")
print(f"PE-Positive: {count_pos} ({count_pos/total*100:.2f}%)")
print(f"PE-Negative: {count_neg} ({count_neg/total*100:.2f}%)")
print(f"Missing patients: {missing}")

# Plot class distribution
plt.bar(["PE-Negative", "PE-Positive"], [count_neg, count_pos],
        color=["skyblue", "lightcoral"])
plt.title("Class Distribution in FUMPE Dataset")
plt.ylabel("Number of Slices")
plt.text(0, count_neg + 10, str(count_neg), ha='center')
plt.text(1, count_pos + 10, str(count_pos), ha='center')
plt.tight_layout()
plt.show()
