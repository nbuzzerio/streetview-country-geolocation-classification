from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Project root = parent of "scripts" directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Processed data directory and results directory
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "best_resnet18.pt"

BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ImageNet normalization stats (must match training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------
# DATASET AND DATALOADER
# ---------------------------------------------------------

def create_test_dataloader(data_dir: Path, batch_size: int):
    test_dir = data_dir / "test"
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_names = test_ds.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes in test set.")
    print(f"Test images: {len(test_ds)}")

    return test_loader, class_names


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def create_model(num_classes: int):
    # Same architecture as in train_model.py
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------
# CONFUSION MATRIX PLOTTING
# ---------------------------------------------------------

def plot_confusion_matrix(cm, class_names, out_path, normalize=False, max_annot_classes=40):
    """
    Plot a confusion matrix and save it to disk.
    If number of classes > max_annot_classes, skip per-cell annotations and axis labels
    to keep the figure readable.
    """
    if normalize:
        cm = cm.astype("float")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    num_classes = len(class_names)
    big_matrix = num_classes > max_annot_classes

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Confusion Matrix")
    plt.colorbar()

    if not big_matrix:
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
        plt.yticks(tick_marks, class_names, fontsize=6)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                if value == 0:
                    continue
                plt.text(
                    j,
                    i,
                    format(value, fmt),
                    horizontalalignment="center",
                    color="white" if value > thresh else "black",
                    fontsize=5,
                )
    else:
        plt.xticks([])
        plt.yticks([])

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved confusion matrix to {out_path}")
    plt.close()


# ---------------------------------------------------------
# MAIN EVALUATION LOGIC
# ---------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            f"Run scripts/train_model.py first to create best_resnet18.pt."
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load test data
    test_loader, class_names = create_test_dataloader(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    # Build model and load trained weights
    model = create_model(num_classes).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_labels = []
    all_preds = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    test_acc = correct / total
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Classification report (precision/recall/F1 per class)
    print("\nClassification report (per class):")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Full confusion matrix (may be hard to read with many classes)
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / "test_confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / "test_confusion_matrix_normalized.png", normalize=True)

    # Confusion matrix for top-20 most frequent classes
    row_sums = cm.sum(axis=1)
    top_k = 20
    top_indices = np.argsort(row_sums)[::-1][:top_k]

    cm_top = cm[np.ix_(top_indices, top_indices)]
    class_names_top = [class_names[i] for i in top_indices]

    plot_confusion_matrix(
        cm_top,
        class_names_top,
        RESULTS_DIR / "test_confusion_matrix_top20.png",
        normalize=True,
        max_annot_classes=40,
    )


if __name__ == "__main__":
    main()
