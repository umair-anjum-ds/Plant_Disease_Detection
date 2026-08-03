"""
model.py — CropDiagnosticCNN
Architecture: MobileNetV3-Small fine-tuned on New Plant Diseases Dataset (Kaggle)
Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
Classes: 11 plant disease / healthy categories
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ── Class labels ──────────────────────────────────────────────────────────────
# IMPORTANT: Must match EXACTLY the folder names in your train/ and valid/ dirs.
# ImageFolder assigns indices alphabetically, so this list must also be alphabetical.
CLASSES = [
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

# Human-readable display names (same order as CLASSES)
DISPLAY_NAMES = [
    "Apple — Black Rot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
    "Corn — Common Rust",
    "Corn — Healthy",
    "Potato — Early Blight",
    "Potato — Late Blight",
    "Potato — Healthy",
    "Tomato — Bacterial Spot",
    "Tomato — Leaf Mold",
    "Tomato — Healthy",
]

# Indices of healthy classes in the alphabetically sorted CLASSES list above:
# Apple___healthy=2, Corn___healthy=4, Potato___healthy=7, Tomato___healthy=10
HEALTHY_INDICES = {2, 4, 7, 10}

NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 11, "Expected 11 classes"


class CropDiagnosticCNN(nn.Module):
    """
    MobileNetV3-Small backbone with two heads:
      - classifier_head : disease class logits      (11 outputs)
      - severity_head   : leaf-damage severity score in [0, 1]

    Training strategy:
      Phase 1 (epochs 1-5)  : backbone frozen, train heads only
      Phase 2 (epochs 6-15) : backbone unfrozen, fine-tune end-to-end
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)

        # Feature extractor + pooling from MobileNetV3
        self.features = base.features
        self.avgpool  = base.avgpool

        in_features = base.classifier[0].in_features  # 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Head 1 — Disease classification
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes),
        )

        # Head 2 — Severity regression (0.0 = healthy, 1.0 = severe disease)
        self.severity_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier_head(x), self.severity_head(x)