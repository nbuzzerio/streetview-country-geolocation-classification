from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Target image size for ResNet-18 input
IMG_SIZE = (224, 224)

# Project root = parent directory of this script's folder
ROOT_DIR = Path(__file__).resolve().parents[1]

# Raw dataset location (HuggingFace OSV5M test subset)
RAW_DIR = ROOT_DIR / "data" / "raw"

# Metadata file from OSV5M
CSV_PATH = RAW_DIR / "test.csv"

# Root directory containing the image shards: 00, 01, 02, 03, 04
RAW_ROOT = RAW_DIR / "images" / "test"

# Output directory for resized/split images and index CSV
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# OSV5M shards (00–04)
SHARD_DIRS = [RAW_ROOT / f"{i:02d}" for i in range(5)]


def build_image_index():
    """
    Walks shard directories (00–04) and builds a mapping of:
        file_stem (image ID as string) -> full image path
    Example:
        "547473234108938" -> Path("images/test/03/547473234108938.jpg")
    """
    index = {}

    for shard in SHARD_DIRS:
        if not shard.exists():
            print(f"Warning: shard {shard} does not exist, skipping.")
            continue

        for img_path in shard.iterdir():
            if not img_path.is_file():
                continue

            stem = img_path.stem  # filename without extension
            index[stem] = img_path

    print(f"Indexed {len(index)} images from shard directories.")
    return index


def main():
    print("Running:", __file__)
    print(f"ROOT_DIR:      {ROOT_DIR}")
    print(f"RAW_DIR:       {RAW_DIR}")
    print(f"CSV_PATH:      {CSV_PATH}")
    print(f"RAW_ROOT:      {RAW_ROOT}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")

    # ---------------------------------------------------------
    # Load metadata CSV
    # ---------------------------------------------------------
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found at: {CSV_PATH}")

    print(f"Loading metadata from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Validate expected columns
    required_cols = ["id", "country"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in test.csv but it was not found.")

    # Convert ID to string to match filename stems
    df["file_stem"] = df["id"].astype(str)

    # ---------------------------------------------------------
    # Build index of image paths
    # ---------------------------------------------------------
    image_index = build_image_index()

    # Keep only rows that map to existing image files
    df = df[df["file_stem"].isin(image_index.keys())].copy()
    print(f"{len(df)} rows match valid image files.")

    # ---------------------------------------------------------
    # Train/Validation/Test split (70/15/15)
    # ---------------------------------------------------------
    print("Performing 70/15/15 random split (no sklearn).")

    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_df = df_shuffled.iloc[:n_train]
    val_df = df_shuffled.iloc[n_train:n_train + n_val]
    test_df = df_shuffled.iloc[n_train + n_val:]

    print(f"Split sizes → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # Create processed output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    index_rows = []

    # ---------------------------------------------------------
    # Process and save resized images
    # ---------------------------------------------------------
    for split_name, split_df in splits.items():
        print(f"\nProcessing split: {split_name} ({len(split_df)} images)")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            stem = row["file_stem"]
            country = row["country"]

            src_path = image_index.get(stem)
            if src_path is None:
                continue

            # Destination folder structure: data/processed/<split>/<country>/
            out_dir = PROCESSED_DIR / split_name / country
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / src_path.name

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img.save(out_path, format="JPEG")

                index_rows.append({
                    "filepath": str(out_path.relative_to(PROCESSED_DIR)),
                    "label": country,
                    "split": split_name,
                })

            except Exception as e:
                print(f"Failed to process {src_path}: {e}")
                continue

    # ---------------------------------------------------------
    # Save processed index CSV
    # ---------------------------------------------------------
    index_df = pd.DataFrame(index_rows)
    index_csv = PROCESSED_DIR / "processed_index.csv"
    index_df.to_csv(index_csv, index=False)

    print("\nCompleted processing.")
    print(f"Processed images saved to: {PROCESSED_DIR}")
    print(f"Index file saved to:       {index_csv}")


if __name__ == "__main__":
    main()
