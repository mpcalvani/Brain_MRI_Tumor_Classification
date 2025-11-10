# Brain MRI Tumor Classification

This project explores automatic classification of brain MRI images into four classes:

- **glioma**
- **meningioma**
- **pituitary tumor**
- **no tumor**

using a public brain MRI dataset.

---

## Dataset

- Brain Tumor MRI dataset with separated `Training` and `Testing` folders.
- Each folder contains 4 labeled subfolders: `glioma`, `meningioma`, `notumor`, `pituitary`.
- The dataset is stored locally under `data/brain_tumor_mri/` and is **not** uploaded to this repository.

---

## Implemented steps

### 1. Data exploration (`notebooks/01_exploration.ipynb`)

- Loads the dataset structure.
- Counts images per class in training and testing sets.
- Visualizes sample MRIs for each class.

### 2. Preprocessing demo (`notebooks/02_preprocessing.ipynb`)

- Converts MR images to grayscale.
- Resizes images to a fixed resolution.
- Normalizes pixel values to [0, 1].
- Shows a visual comparison before/after preprocessing.

### 3. Baseline classifier (`notebooks/03_model.py`)

- Loads all images (64x64 grayscale, normalized).
- Flattens images into feature vectors.
- Trains a **Random Forest** classifier for 4-way classification.
- Evaluates performance with:
  - classification report (precision, recall, F1-score)
  - confusion matrix (saved as `notebooks/confusion_matrix.png`).

---

## Notes

- This is a **baseline educational project**, not a clinical decision tool.
- Future improvements:
  - convolutional neural networks (CNNs),
  - more advanced preprocessing and data augmentation,
  - cross-validation and external validation datasets.
