# Crop Disease Classifier 🌿

> **Devsinc AI/ML Internship — 3-Day CNN Project**

A plant-leaf disease classifier powered by a fine-tuned **MobileNetV3-Small** CNN, served through a FastAPI backend and a plain HTML/JS frontend. Upload a leaf photo and get an instant diagnosis with treatment recommendation and Grad-CAM heatmap.

**GitHub:** [umair-anjum-ds/Plant_Disease_Detection](https://github.com/umair-anjum-ds/Plant_Disease_Detection)

---

## Goal

Classify a leaf photograph into one of **11 categories** across 4 crops:

| Crop   | Classes Predicted                                     |
|--------|-------------------------------------------------------|
| Apple  | Black Rot · Cedar Apple Rust · Healthy                |
| Corn   | Common Rust · Healthy                                 |
| Potato | Early Blight · Late Blight · Healthy                  |
| Tomato | Bacterial Spot · Leaf Mold · Healthy                  |

**Dataset:** [New Plant Diseases Dataset — Kaggle (vipoooool)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
20,733 train images · 5,183 validation images · 11 classes · balanced (~1,700–2,000 images per class)

---

## Done ✅

### Day 1 — Data and first model

- Downloaded and explored the New Plant Diseases Dataset from Kaggle
- Verified class balance: all 11 classes have 1,700–2,000 images each
- Verified zero data leakage: **0 shared filenames** between train and valid
- Built `model.py`: MobileNetV3-Small backbone with two output heads:
  - Classifier head: 11-class disease prediction
  - Severity head: auxiliary regression (0.0 = healthy, 1.0 = severe)
- Built `train.py`: full pipeline with augmentation, train/val/test split, checkpoint saving
- First model ran end-to-end successfully

### Day 2 — Model tuning and evaluation

- Implemented 2-phase fine-tuning:
  - Phase 1 (epochs 1–5): backbone frozen, heads trained only
  - Phase 2 (epochs 6–15): backbone unfrozen, end-to-end fine-tuning at lower LR
- Augmentation: random flips, rotation ±20°, colour jitter, affine translate
- Deterministic class-balanced 50/50 split of valid → val + test (SEED=42)
- Implemented real Grad-CAM in `gradcam.py` — hooks on `features[-1]` of MobileNetV3
- Saved best checkpoint by val accuracy, confusion matrix, training curves, per-class report

### Day 3 — Web app

- Built `main.py`: FastAPI backend with 4 endpoints:
  - `POST /predict` — returns diagnosis, top-3, severity, Grad-CAM, treatment, latency
  - `GET /classes` — lists all 11 classes
  - `GET /model-info` — returns architecture, accuracy, dataset info
  - `GET /` — health check
- Built `frontend/index.html`: single-file UI, no build step required:
  - Drag-and-drop image upload
  - Confidence bar with colour coding
  - Top-3 predictions with probability bars
  - Severity indicator
  - Grad-CAM heatmap display
  - Disease-specific treatment recommendation
  - Inference latency display

---

## Results

| Metric            | Value                                      |
|-------------------|--------------------------------------------|
| Train Accuracy    | ~97.3%                                     |
| Val Accuracy      | ~98.3%                                     |
| Test Accuracy     | see `artifacts/training_meta.json`         |
| Confusion Matrix  | `artifacts/confusion_matrix.png`           |
| Training Curves   | `artifacts/training_curves.png`            |
| Inference Latency | ~50–150ms per image (CPU)                  |

### Data leakage audit

| Check | Result |
|---|---|
| Train vs valid filename overlap | 0 shared filenames ✅ |
| Val vs test index overlap | 0 shared indices ✅ |
| Val + test covers full valid set | All images accounted for ✅ |
| Augmentation on val/test | None — resize + normalize only ✅ |
| Class balance across val and test | All 11 classes equal ✅ |

### Most common misclassifications

- **Potato Early Blight ↔ Late Blight** — both show dark lesions on mature leaves
- **Tomato Bacterial Spot ↔ Tomato Leaf Mold** — both appear in greenhouse tomatoes

---

## What I Tried That Did NOT Work

| What I tried | What happened | Fix applied |
|---|---|---|
| `labels % 2 == 1` for severity labels | Wrong healthy/disease mapping for our class order | Replaced with explicit `HEALTHY_INDICES = {2, 4, 7, 10}` |
| Double underscore `__` in CLASSES | Mismatch with actual `___` folder names — model predicted everything as Tomato Healthy | Updated all CLASSES to match exact folder names |
| LR = 1e-2 for backbone in Phase 2 | Training collapsed, loss spiked | Fixed to LR/10 for backbone, full LR for heads |
| Fake pixel-colour heuristic (original code) | Always predicted the same class regardless of input | Removed entirely — all predictions now from model softmax |

---

## Currently Working On

- Comparing from-scratch 3-block CNN vs MobileNetV3 fine-tune
- Testing dropout 0.2 vs 0.4 on the classifier head
- Measuring exact impact of colour-jitter augmentation on val accuracy

---

## Future Additions

- Out-of-scope detection: reject images that are not plant leaves
- Side-by-side model comparison in the UI
- Deploy to Hugging Face Spaces
- Before/after augmentation gallery in the frontend
- Per-class F1 score displayed in the UI
- Grad-CAM guided augmentation on under-activated regions

---

## How to Run (From a Fresh Clone)

### 1. Clone the repo

```bash
git clone https://github.com/umair-anjum-ds/Plant_Disease_Detection.git
cd Plant_Disease_Detection
```

### 2. Download the dataset

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/
```

Keep only these 11 folders inside both `data/train/` and `data/valid/`:

```
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Corn_(maize)___Common_rust_
Corn_(maize)___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato___Bacterial_spot
Tomato___Leaf_Mold
Tomato___healthy
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py --data_dir ../data --epochs 15 --batch_size 32
```

Outputs saved to `backend/artifacts/`:

```
best_model.pth         ← best checkpoint by val accuracy
training_meta.json     ← accuracy numbers and class list
confusion_matrix.png   ← per-class confusion matrix on test set
training_curves.png    ← loss and accuracy curves over epochs
```

### 5. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Confirm this line appears at startup:
```
Loaded model from ./artifacts/best_model.pth  (device=cpu)
```

API docs: `http://localhost:8000/docs`

### 6. Open the frontend

```powershell
cd ../frontend
python -m http.server 3000
# Open: http://localhost:3000
```

---

## Project Structure

```
Plant_Disease_Detection/
├── README.md
├── backend/
│   ├── model.py               # CropDiagnosticCNN (MobileNetV3-Small)
│   ├── train.py               # Full training pipeline
│   ├── gradcam.py             # Real Grad-CAM implementation
│   ├── main.py                # FastAPI API server
│   ├── requirements.txt
│   └── artifacts/             # Created by train.py
│       ├── best_model.pth
│       ├── training_meta.json
│       ├── confusion_matrix.png
│       └── training_curves.png
└── frontend/
    └── index.html             # Self-contained UI (no build step)
```

---

## .gitignore

```
backend/artifacts/best_model.pth
data/
__pycache__/
*.pyc
.env
venv/
```

---

*Devsinc AI/ML Internship · 3-Day CNN Project · New Plant Diseases Dataset (Kaggle)*