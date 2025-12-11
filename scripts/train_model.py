import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm
import itertools


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Project root = parent of the "scripts" directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Processed data directory produced by make_splits.py
DATA_DIR = BASE_DIR / "data" / "processed"

NUM_EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------
# DATASETS AND DATALOADERS
# ---------------------------------------------------------

def create_dataloaders(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)

    # Use num_workers=0 to avoid multiprocessing issues on some Windows setups
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_names = train_ds.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes.")
    print(f"Train: {len(train_ds)} images")
    print(f"Val:   {len(val_ds)} images")
    print(f"Test:  {len(test_ds)} images")

    return train_loader, val_loader, test_loader, class_names


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def create_model(num_classes: int):
    # Load ResNet-18 with pretrained ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final fully connected layer to match our num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------
# TRAINING AND EVALUATION LOOPS
# ---------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", unit="batch", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", unit="batch", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return epoch_loss, epoch_acc, all_labels, all_preds


# ---------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------

def plot_curves(train_values, val_values, ylabel, out_path):
    epochs = np.arange(1, len(train_values) + 1)

    plt.figure()
    plt.plot(epochs, train_values, marker="o", label="Train")
    plt.plot(epochs, val_values, marker="o", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved {ylabel} curve to {out_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path, normalize=False, max_annot_classes=40):
    """
    If number of classes > max_annot_classes, we skip per-cell text annotations
    and axis labels to keep the figure readable.
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
    plt.title("Confusion Matrix")
    plt.colorbar()

    if not big_matrix:
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
        plt.yticks(tick_marks, class_names, fontsize=6)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
# MAIN
# ---------------------------------------------------------

def main(dry_run: bool = False):
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)

    model = create_model(num_classes).to(DEVICE)

    if dry_run:
        # Smoke test: single forward pass on one batch
        try:
            images, labels = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("Train loader is empty. Ensure processed data exists in data/processed.")

        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        print(f"Forward pass OK, output shape: {tuple(outputs.shape)}")
        return

    # Full training run
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_labels, val_preds = eval_one_epoch(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = results_dir / "best_resnet18.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} (val_acc={val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes. Best val acc: {best_val_acc:.4f}")

    # Plot curves
    plot_curves(train_losses, val_losses, "Loss", results_dir / "loss_curve.png")
    plot_curves(train_accs, val_accs, "Accuracy", results_dir / "accuracy_curve.png")

    # Confusion matrix on validation set for final epoch
    cm = confusion_matrix(val_labels, val_preds)

    # Full confusion matrices
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, class_names, results_dir / "confusion_matrix_normalized.png", normalize=True)

    # Confusion matrix for top-20 most frequent classes
    row_sums = cm.sum(axis=1)
    top_k = 20
    top_indices = np.argsort(row_sums)[::-1][:top_k]

    cm_top = cm[np.ix_(top_indices, top_indices)]
    class_names_top = [class_names[i] for i in top_indices]

    plot_confusion_matrix(
        cm_top,
        class_names_top,
        results_dir / "confusion_matrix_top20.png",
        normalize=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single forward pass on one batch and exit.",
    )
    args = parser.parse_args()

    main(dry_run=args.dry_run)
