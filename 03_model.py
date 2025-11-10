import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("Inizio esecuzione 03_model.py")

# 1. Percorsi e parametri
base_dir = "../data/brain_tumor_mri"
train_dir = os.path.join(base_dir, "Training")
test_dir = os.path.join(base_dir, "Testing")

classes = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (64, 64)  # più piccolo = più veloce

def load_images(root_dir):
    X = []
    y = []
    for label, cls in enumerate(classes):
        folder = os.path.join(root_dir, cls)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath).convert("L")             # grayscale
                img = img.resize(IMAGE_SIZE)                    # resize
                arr = np.array(img, dtype=np.float32) / 255.0   # normalizza 0-1
                X.append(arr.flatten())                         # 64x64 -> 4096
                y.append(label)
            except Exception as e:
                # se un file è rovinato, lo saltiamo
                print(f"Skipped {fpath}: {e}")
                continue
    return np.array(X), np.array(y)

print("Carico immagini di TRAIN...")
X_train, y_train = load_images(train_dir)
print("Carico immagini di TEST...")
X_test, y_test = load_images(test_dir)

print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# 2. Modello Random Forest
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("Alleno il modello...")
clf.fit(X_train, y_train)
print("Modello allenato.")

# 3. Valutazione
y_pred = clf.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred, target_names=classes))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# 4. Salva una confusion matrix come immagine
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
plt.yticks(ticks=range(len(classes)), labels=classes)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion matrix salvata come confusion_matrix.png")