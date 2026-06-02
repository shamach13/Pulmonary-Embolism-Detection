# Pulmonary Embolism Detection using CNN and Conditional GAN Augmentation

## Overview

Pulmonary Embolism (PE) is a potentially life-threatening condition caused by blockage of pulmonary arteries. Early and accurate detection from Computed Tomography Pulmonary Angiography (CTPA) scans is critical for timely diagnosis and treatment.

This project proposes an explainable deep learning framework for PE detection using:

* Convolutional Neural Network (CNN) classifier
* Conditional Generative Adversarial Network (cGAN) for data augmentation
* Grad-CAM based explainability
* Performance evaluation using ROC curves, confusion matrices, and classification metrics

The objective is to address class imbalance in PE datasets by generating synthetic PE-positive samples and improving model performance.

---

## Dataset

**FUMPE (Ferdowsi University of Mashhad Pulmonary Embolism Dataset)**

The dataset consists of CTPA scans with expert-annotated pulmonary embolism regions.

Dataset is not included in this repository due to licensing and storage constraints.

---

## Methodology

### 1. Preprocessing

* Extraction of CTPA slices
* Ground truth overlay generation
* Image resizing to 128 × 128
* Intensity normalization

### 2. Class Imbalance Handling

* Extraction of PE-positive slices
* Synthetic sample generation using cGAN

### 3. CNN-based PE Classification

The classifier consists of:

* Convolutional layers
* ReLU activations
* Max-pooling layers
* Fully connected classification layer
* Sigmoid output for binary classification

### 4. Explainability

Grad-CAM is used to visualize discriminative image regions responsible for PE predictions.

---

## Project Workflow

CTPA Dataset (FUMPE)
→ Preprocessing
→ PE-positive Slice Extraction
→ cGAN Augmentation
→ Balanced Training Dataset
→ CNN Classification
→ PE / Non-PE Prediction
→ Grad-CAM Explainability

---

## Experimental Results

### Baseline CNN

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 75.13% |
| Recall   | 21.16% |
| F1 Score | 30.80% |
| AUC      | 0.803  |

### CNN + cGAN Augmentation

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 74.91% |
| Recall   | 46.38% |
| F1 Score | 49.16% |
| AUC      | 0.814  |

The cGAN-augmented model significantly improves recall and F1-score while achieving higher AUC.

---

## Repository Structure

```text
Pulmonary-Embolism-Detection/
│
├── src/
│   ├── check_balance.py
│   ├── preprocess_fumpe.py
│   ├── dataset_split.py
│   ├── pe_classifier_baseline.py
│   ├── pe_classifier_with_cgan_aug.py
│   ├── test_evaluation.py
│   └── gradcam_pe.py
│
├── figures/
│   ├── check_balance.png
│   ├── preprocessing.png
|   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── training_accuracy_curve.png
│   ├── training_loss_curve.png
│   └── gradcam_comparison.png
│
├── models/
│   ├── pe_baseline.pth
│   └── pe_classifier_cgan_aug.pth
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shamach13/Pulmonary-Embolism-Detection.git
cd Pulmonary-Embolism-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Preprocess Dataset

```bash
python preprocess_fumpe.py
```

### Train Baseline CNN

```bash
python pe_classifier_baseline.py
```

### Train CNN with Augmented Dataset

```bash
python pe_classifier_with_cgan_aug.py
```

### Evaluate Model

```bash
python test_evaluation.py
```

### Generate ROC Curve


### Generate Grad-CAM Visualizations

```bash
python gradcam_pe.py
```

---



## Author

**Shama Chandukudlu**

M.Tech Computer Science and Engineering
Manipal Institute of Technology, Manipal

