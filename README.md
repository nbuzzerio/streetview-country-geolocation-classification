# StreetView Country Geolocation Classification

### Country-Level Geolocation from Street-View Imagery Using a Pretrained ResNet-18 Model

This repository contains the full codebase, documentation, and project structure for a deep learning project that predicts the **country** of a street-view image.  
The model uses a **pretrained ResNet-18** backbone and includes data preprocessing, training, evaluation metrics, and reproducibility instructions.

This repository is structured for **academic submission** and **easy reproduction** by instructors or reviewers.

---

## Repository Structure

```
streetview-country-geolocation-classification/
├─ README.md
├─ LICENSE
├─ requirements.txt  (or pyproject.toml)
├─ .gitignore
├─ .env.example
│
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
│
├─ notebooks/
│  ├─ 01_exploration.ipynb
│  ├─ 02_preprocessing.ipynb
│  └─ 03_modeling.ipynb
│
├─ src/
│  ├─ config/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ evaluation/
│  └─ utils/
│
├─ scripts/
│  ├─ make_splits.py
│  ├─ train_model.py
│  ├─ evaluate_model.py
│  └─ visualize_results.py
│
├─ tests/
│  ├─ test_data_utils.py
│  ├─ test_metrics.py
│  └─ test_models.py
│
├─ reports/
│  ├─ figures/
│  └─ final_report/
│
└─ docs/
   ├─ architecture.md
   ├─ data_card.md
   └─ usage.md
```

---

## Getting Started

### 1. Clone the Repository

```
git clone https://github.com/nbuzzerio/streetview-country-geolocation-classification.git
cd streetview-country-geolocation-classification
```

---

## Python Environment Setup

### 2. Create a Virtual Environment

```
python -m venv .venv
```

Activate it:

**Windows**

```
.venv\Scripts\activate
```

**Mac/Linux**

```
source .venv/bin/activate
```

---

## Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

This project relies on the following major Python libraries:

- **Python 3.10+**
- **PyTorch** — deep learning framework used to train the ResNet-18 model  
- **Torchvision** — pretrained models and image transform utilities  
- **Pandas** — metadata loading and preprocessing  
- **NumPy** — numerical operations  
- **scikit-learn** — confusion matrix and classification metrics  
- **Pillow (PIL)** — image loading and resizing  
- **Matplotlib** — plotting accuracy/loss curves and confusion matrices  
- **tqdm** — progress bars for preprocessing and training loops  

All dependencies are listed in:

```
requirements.txt
```

Install them with:

```
pip install -r requirements.txt
```


---

## Dataset Placement

Place your dataset inside:

```
data/raw/
```

Example:

```
data/raw/images/...
data/raw/metadata.csv
```

---

## Dataset

This project uses the **OpenStreetView (OSV5M)** dataset, which contains over five million geolocated street-view images labeled with GPS coordinates and country information. For this project, we specifically use the **210,000-image test subset** provided on HuggingFace.

### Downloading the Dataset

The OSV5M dataset is hosted on HuggingFace:

https://huggingface.co/datasets/osv5m/osv5m

HuggingFace may update hosting paths or download instructions over time.  
For this reason, **please refer to the dataset page directly** for the most accurate and up-to-date download method.

This project uses the **test subset**, which consists of several `.zip` files under the `images/test` directory on HuggingFace.
After downloading and extracting these files, place them into:

```
data/raw/
```

## Expected Directory Structure

```
data/raw/
├─ images/
│ └─ test/
│ ├─ 00/
│ │ ├─ 00001.jpg
│ │ ├─ 00002.jpg
│ │ └─ ...
│ ├─ 01/
│ ├─ 02/
│ ├─ 03/
│ └─ 04/
└─ test.csv

```

---

## Required Columns in `test.csv`

Your preprocessing pipeline expects the following columns:

- `image_path` — relative path to each image (e.g., `images/test/00/00001.jpg`)
- `country` — country label used for classification
- `latitude`
- `longitude`
- `land_cover` (optional)
- `climate` (optional)
- `region` (optional)

---

## Notes

- The dataset is **not included in this repository** due to size constraints.
- Only the **test subset (≈210k images)** is required for this project.
- Preprocessing scripts assume the images and metadata file are placed inside `data/raw/` exactly as shown.

---

## Running the Project

### 1. Create Train/Validation/Test Splits

```
python scripts/make_splits.py
```

Processed data will appear in:

```
data/processed/
```

---

### 2. Run a Smoke Test (Forward Pass Only)

```
python scripts/train_model.py --dry-run
```

Expected output:

```
Forward pass OK, output shape: (batch_size, num_classes)
```

---

### 3. Train the Full Model

```
python scripts/train_model.py
```

---

### 4. Evaluate Model Performance

```
python scripts/evaluate_model.py
```

Metrics include:

- Accuracy
- Precision, Recall, F1
- Top-k accuracy
- Haversine distance
- Confusion matrix

---

## Running Tests

```
pytest
```

---

## Reports

Generated figures appear in:

```
reports/figures/
```

Final written report goes in:

```
reports/final_report/
```

---

## Reproducibility Notes

This project was validated by:

1. Pushing to GitHub
2. Downloading the repository as a ZIP
3. Following this README step-by-step on a clean environment
4. Verifying all scripts run successfully

This ensures the project is fully reproducible for instructors.

---

## License

MIT License

---

## Contact

Nicholas Buzzerio  
MSAI — Dakota State University  
GitHub: https://github.com/nbuzzerio
