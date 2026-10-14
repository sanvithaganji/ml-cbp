"""
Unit Tests for Inference Pipeline and Explainability.
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image

from src.models.image_model import LeafDiseaseEfficientNet
from src.explainability.gradcam import GradCAM


def test_gradcam_generation():
    model = LeafDiseaseEfficientNet(num_classes=3, pretrained=False)
    gradcam = GradCAM(model)

    x = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(x, target_class=1)

    assert heatmap.shape == (7, 7) or heatmap.shape == (224, 224) or len(heatmap.shape) == 2
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    dummy_pil = Image.new("RGB", (224, 224), color=(34, 139, 34))
    overlay = gradcam.overlay_on_image(dummy_pil, heatmap)
    assert overlay.size == (224, 224)
