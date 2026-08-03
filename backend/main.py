"""
main.py — FastAPI backend for the Crop Disease Classifier
Loads a TRAINED CropDiagnosticCNN checkpoint and serves honest predictions.

Dataset : New Plant Diseases Dataset (Kaggle)
          https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Run:
    uvicorn main:app --reload --port 8000
"""

import io
import base64
import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import CropDiagnosticCNN, CLASSES, DISPLAY_NAMES, NUM_CLASSES
from gradcam import GradCAM

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Classifier API",
    description=(
        "Classifies plant leaf images into 11 categories (disease / healthy) "
        "using a MobileNetV3-Small CNN fine-tuned on the New Plant Diseases Dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Treatment recommendations ─────────────────────────────────────────────────
TREATMENTS = {
    "Apple___Black_rot": (
        "Remove mummified fruits and infected wood immediately. "
        "Apply captan or thiophanate-methyl fungicide during the growing season. "
        "Prune dead branches and dispose of fallen leaves."
    ),
    "Apple___Cedar_apple_rust": (
        "Apply fungicides (myclobutanil or trifloxystrobin) from pink bud stage onwards. "
        "Remove nearby juniper or cedar trees if possible as they host the alternate stage of the fungus."
    ),
    "Apple___healthy": (
        "No disease detected. Maintain regular pruning, balanced fertilisation, "
        "and monitor for early signs of scab or rust."
    ),
    "Corn_(maize)___Common_rust_": (
        "Apply foliar fungicides (triazoles or strobilurins) at early rust detection. "
        "Plant rust-tolerant hybrids where available. Scout fields regularly."
    ),
    "Corn_(maize)___healthy": (
        "No disease detected. Continue normal irrigation and crop rotation practices."
    ),
    "Potato___Early_blight": (
        "Use chlorothalonil or copper-based fungicide on a 7-10 day schedule. "
        "Ensure adequate plant spacing for airflow and avoid overhead irrigation."
    ),
    "Potato___Late_blight": (
        "Remove and destroy infected foliage immediately — do not compost. "
        "Apply mancozeb or metalaxyl fungicide. Destroy nearby volunteer plants."
    ),
    "Potato___healthy": (
        "No disease detected. Monitor regularly; "
        "late blight spreads rapidly in wet and cool conditions."
    ),
    "Tomato___Bacterial_spot": (
        "Apply copper-based bactericide combined with mancozeb. "
        "Avoid overhead watering. Remove and destroy severely infected plants."
    ),
    "Tomato___Leaf_Mold": (
        "Improve greenhouse ventilation to keep relative humidity below 85%. "
        "Remove infected lower leaves. Apply chlorothalonil if symptoms are severe."
    ),
    "Tomato___healthy": (
        "No disease detected. Maintain consistent watering and "
        "watch for early symptom signs on lower leaves."
    ),
}

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "./artifacts/best_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training metadata if available
META_PATH     = "./artifacts/training_meta.json"
training_meta: dict = {}
if Path(META_PATH).exists():
    with open(META_PATH) as f:
        training_meta = json.load(f)


def _load_model() -> CropDiagnosticCNN:
    """Load model once at startup. Raises a clear error if checkpoint is missing."""
    model = CropDiagnosticCNN(num_classes=NUM_CLASSES, freeze_backbone=False)
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at '{MODEL_PATH}'. "
            f"Run train.py first to produce a real model before serving predictions."
        )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}  (device={DEVICE})")
    return model


model: CropDiagnosticCNN = _load_model()
gradcam = GradCAM(model, target_layer=model.features[-1])

# ── Image preprocessing (must match training val_tf exactly) ──────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def _encode_bgr_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":         "ok",
        "project":        "Crop Disease Classifier",
        "classes":        DISPLAY_NAMES,
        "model_accuracy": training_meta.get("test_accuracy"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a leaf image (JPEG / PNG / WEBP).
    Returns:
      - diagnosis       : human-readable class name
      - confidence      : softmax probability for top class (real, from model)
      - top3            : top-3 predictions with probabilities
      - severity_pct    : auxiliary severity head output
      - treatment       : disease-specific recommendation
      - gradcam_heatmap : base64 JPEG of Grad-CAM overlay
      - inference_ms    : server-side inference latency
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image (JPEG/PNG/WEBP).")

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    # ── Inference ─────────────────────────────────────────────────────────────
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

    t_start = time.perf_counter()
    with torch.no_grad():
        logits, sev_tensor = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]   # shape: (NUM_CLASSES,)

    inference_ms = round((time.perf_counter() - t_start) * 1000, 1)

    top3_vals, top3_idx = probs.topk(3)
    pred_idx   = int(top3_idx[0].item())
    confidence = float(top3_vals[0].item())
    severity   = float(sev_tensor[0][0].item())

    top3 = [
        {
            "class":       DISPLAY_NAMES[int(i)],
            "probability": round(float(p), 4),
        }
        for p, i in zip(top3_vals, top3_idx)
    ]

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    cam = gradcam.generate(
        PREPROCESS(pil_img).unsqueeze(0).to(DEVICE),
        class_idx=pred_idx,
    )
    overlay_bgr = GradCAM.overlay(pil_img, cam)
    gradcam_b64 = _encode_bgr_to_b64(overlay_bgr)

    # ── Response ──────────────────────────────────────────────────────────────
    class_key = CLASSES[pred_idx]
    return {
        "diagnosis":       DISPLAY_NAMES[pred_idx],
        "confidence":      round(confidence * 100, 2),
        "top3":            top3,
        "severity_pct":    round(severity * 100, 1),
        "treatment":       TREATMENTS[class_key],
        "gradcam_heatmap": gradcam_b64,
        "inference_ms":    inference_ms,
    }


@app.get("/classes")
def list_classes():
    return {"classes": DISPLAY_NAMES, "count": NUM_CLASSES}


@app.get("/model-info")
def model_info():
    info = {
        "architecture": "MobileNetV3-Small (fine-tuned)",
        "dataset":      "New Plant Diseases Dataset (Kaggle)",
        "dataset_url":  "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
        "num_classes":  NUM_CLASSES,
        "input_size":   "224x224",
        "device":       str(DEVICE),
    }
    for k in ("test_accuracy", "best_val_accuracy", "epochs_trained"):
        if k in training_meta:
            info[k] = training_meta[k]
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)