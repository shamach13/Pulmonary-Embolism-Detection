import os
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = "/home/user30/SHAMA_Research/FUMPE/Processed"
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

files = sorted(os.listdir(IMG_DIR))

labels = [np.load(os.path.join(LABEL_DIR, f)) for f in files]

# Train (70%) + temp (30%)
train_files, temp_files, train_labels, temp_labels = train_test_split(
    files, labels, test_size=0.3, stratify=labels, random_state=42
)

# Val (15%) + Test (15%)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

np.save("train_files.npy", train_files)
np.save("val_files.npy", val_files)
np.save("test_files.npy", test_files)

print("Split done:")
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))
