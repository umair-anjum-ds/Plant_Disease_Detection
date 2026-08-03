"""
gradcam.py — Real Grad-CAM implementation for CropDiagnosticCNN
Hooks into the last convolutional block of MobileNetV3-Small (features[-1])
and produces a class-discriminative heatmap.
"""

import cv2
import numpy as np
import torch


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
    Works with the MobileNetV3-Small backbone in CropDiagnosticCNN.

    Usage:
        gcam = GradCAM(model, target_layer=model.features[-1])
        cam  = gcam.generate(input_tensor, class_idx=None)   # None → predicted class
        overlay = gcam.overlay(original_pil_image, cam)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Returns a float32 numpy array in [0, 1] of shape (H, W)
        where H=W=224 (the input spatial size).
        """
        self.model.eval()
        # Enable gradients for this forward pass only
        input_tensor = input_tensor.requires_grad_(True)

        logits, _ = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pooling of gradients → channel weights
        gradients   = self._gradients.cpu().numpy()[0]    # (C, H', W')
        activations = self._activations.cpu().numpy()[0]  # (C, H', W')

        weights = gradients.mean(axis=(1, 2))             # (C,)
        cam = np.einsum("c,chw->hw", weights, activations)

        # ReLU + normalise
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    @staticmethod
    def overlay(pil_img, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blends a BGR OpenCV image with the Grad-CAM heatmap.
        Returns a uint8 BGR numpy array.
        """
        img_bgr = cv2.cvtColor(np.array(pil_img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        heatmap  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        blended  = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
        return blended
