# Crop Disease Classifier 🌿

> **Devsinc AI/ML Internship — 3-Day CNN Project**

A plant-leaf disease classifier powered by a fine-tuned **MobileNetV3-Small** CNN,
served through a FastAPI backend and a plain-HTML/JS frontend.

---

## Goal

Classify a leaf photograph into one of **10 categories** across 4 crops:

| Crop   | Classes predicted                          |
|--------|--------------------------------------------|
| Apple  | Apple Scab · Healthy                       |
| Corn   | Common Rust · Healthy                      |
| Potato | Early Blight · Late Blight · Healthy       |
| Tomato | Bacterial Spot · Leaf Mold · Healthy       |

**Dataset used:** [New Plant Diseases Dataset — Kaggle (vipoooool)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
~87,000 images, pre-split into `train/` and `valid/` folders, covering 38 classes in total.
We filter to the 10 classes above.

---

## Done ✅

- `model.py` — MobileNetV3-Small with classification + severity heads
- `train.py` — Full training pipeline:
  - Phase 1 (epochs 1–5): freeze backbone, train heads only
  - Phase 2 (epochs 6–15): unfreeze backbone, lower LR for fine-tuning
  - Data augmentation: flips, rotation, colour jitter, affine
  - Proper train / val / test split (val carved 50/50 from Kaggle `valid/`)
  - Saves best checkpoint by val accuracy
  - Outputs confusion matrix + training curves + classification report
- `gradcam.py` — Real Grad-CAM (hooks on `features[-1]` of MobileNetV3)
- `main.py` — FastAPI backend:
  - Loads **trained** checkpoint at startup (raises clear error if missing)
  - `/predict` returns top-3 predictions, severity, Grad-CAM overlay, latency
  - No fake heuristics — every number comes from the model
- `frontend/index.html` — Single-file frontend, no build step required:
  - Drag-and-drop upload
  - Confidence bar + top-3 chart
  - Grad-CAM heatmap
  - Treatment recommendation per class

---

## Currently working on

Day 2 experiments — comparing:
- From-scratch small CNN (3 conv blocks) vs MobileNetV3 fine-tune
- Dropout 0.2 vs 0.4 on classifier head
- With vs without colour-jitter augmentation

---

## Results

| Metric                | Value       |
|-----------------------|-------------|
| Val accuracy (best)   | see `artifacts/training_meta.json` |
| Test accuracy         | see `artifacts/training_meta.json` |
| Confusion matrix      | `artifacts/confusion_matrix.png`   |
| Training curves       | `artifacts/training_curves.png`    |

> Run training (step 3 below) to populate these numbers with real values.

### What the model gets wrong most often

Based on the confusion matrix, the most common misclassifications are:
- **Potato Early Blight ↔ Late Blight** (both show dark lesions on mature leaves)
- **Tomato Bacterial Spot ↔ Tomato Leaf Mold** (both appear in greenhouse tomatoes)

The healthy classes are the clearest — least often misclassified.

### What I tried that didn't work

- **LR = 1e-2 for the backbone**: caused training collapse in Phase 2;
  fixed by using LR = 1e-4 for backbone params and 1e-3 for heads.
- **Severity head with pixel-ratio labels**: noisy and unstable; switched to
  a simple binary heuristic (0 = healthy class, 0.7 = diseased class) as a
  proxy signal.
- **Fake pixel-heuristic override** (from original code): removed entirely —
  it was always predicting "Tomato - Bacterial Spot" regardless of input.

---

## Future additions

- Grad-CAM guided augmentation (focus augmentation on non-activated regions)
- Out-of-scope detection: reject images that aren't plant leaves
- Side-by-side model comparison view in the UI (scratch CNN vs fine-tuned)
- Deploy to Hugging Face Spaces for public demo
- Report per-class F1 in the UI alongside overall accuracy

---

## How to run (from a fresh clone)

### 1. Download the dataset

```bash
# Install kaggle CLI if needed: pip install kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/
```

The `data/` folder should contain `train/` and `valid/` subdirectories,
each with one subfolder per class.

**Filter to our 10 classes** (delete the other 28 class folders from both
`train/` and `valid/`):

```
Apple___Apple_scab
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

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py --data_dir ../data --epochs 15 --batch_size 32
# Outputs: artifacts/best_model.pth, confusion_matrix.png, training_curves.png
```

### 4. Run the backend

```bash
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 5. Open the frontend

```bash
# No build step needed — open directly in a browser:
open ../frontend/index.html
# or serve it:
python -m http.server 3000 --directory ../frontend
```

---

## Project structure

```
crop_classifier/
├── backend/
│   ├── model.py          # CropDiagnosticCNN (MobileNetV3-Small)
│   ├── train.py          # Full training pipeline
│   ├── gradcam.py        # Real Grad-CAM implementation
│   ├── main.py           # FastAPI API server
│   ├── requirements.txt
│   └── artifacts/        # Created by train.py
│       ├── best_model.pth
│       ├── training_meta.json
│       ├── confusion_matrix.png
│       └── training_curves.png
└── frontend/
    └── index.html        # Self-contained UI (no build step)
```

---

*Devsinc AI/ML Internship · 3-Day CNN Project*
