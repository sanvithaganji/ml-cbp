"""
Multimodal Fusion Architecture.
Implements:
1. Feature-Level Fusion (MultimodalFusionNet):
   Visual backbone (EfficientNet-B0) + Sensor MLP -> Embedding Projection
   -> Concatenation -> Fusion Classification Head.
2. Late Fusion (LateFusionClassifier):
   Combines calibrated probability outputs of image and sensor models.
"""

from typing import Tuple, Optional, Dict, Any
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultimodalFusionNet(nn.Module):
    """
    Feature-level multimodal fusion network.
    Combines visual representations from leaf images with 5 environmental measurements.
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_sensor_features: int = 5,
        pretrained: bool = True,
        image_embed_dim: int = 128,
        sensor_embed_dim: int = 32,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_embed_dim = image_embed_dim
        self.sensor_embed_dim = sensor_embed_dim
        self.fused_dim = image_embed_dim + sensor_embed_dim

        # 1. Visual Modality Branch (EfficientNet-B0 backbone)
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.visual_backbone = backbone.features
        self.visual_pool = backbone.avgpool
        self.raw_visual_dim = 1280

        # Visual Projection Head (1280 -> 128)
        self.visual_projector = nn.Sequential(
            nn.Linear(self.raw_visual_dim, image_embed_dim),
            nn.LayerNorm(image_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # 2. Environmental Modality Branch (5 -> 64 -> 32)
        self.sensor_branch = nn.Sequential(
            nn.Linear(num_sensor_features, 64),
            nn.ReLU(),
            nn.Linear(64, sensor_embed_dim),
            nn.LayerNorm(sensor_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # 3. Multimodal Fusion Head (160 -> 64 -> num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def extract_modal_embeddings(
        self,
        image: torch.Tensor,
        sensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract projected embeddings for both modalities independently.
        """
        v_feat = self.visual_backbone(image)
        v_pooled = torch.flatten(self.visual_pool(v_feat), 1)
        v_embed = self.visual_projector(v_pooled)
        s_embed = self.sensor_branch(sensor)
        return v_embed, s_embed

    def forward(
        self,
        image: torch.Tensor,
        sensor: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through multimodal pipeline.
        Args:
            image: Image tensor of shape (B, 3, 224, 224)
            sensor: Sensor tensor of shape (B, 5)
            return_embeddings: If True, returns (logits, v_embed, s_embed)
        Returns:
            logits: (B, num_classes)
        """
        v_embed, s_embed = self.extract_modal_embeddings(image, sensor)
        # Feature-level concatenation
        fused = torch.cat([v_embed, s_embed], dim=1)
        logits = self.fusion_head(fused)

        if return_embeddings:
            return logits, v_embed, s_embed
        return logits


class LateFusionClassifier:
    """
    Late fusion module combining predicted probability vectors from
    an image model and a sensor model.
    P_late = w_img * P_img + (1 - w_img) * P_sensor
    """

    def __init__(self, image_weight: float = 0.80):
        self.image_weight = max(0.0, min(1.0, float(image_weight)))
        self.sensor_weight = 1.0 - self.image_weight

    def combine_probabilities(
        self,
        image_probs: torch.Tensor,
        sensor_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blends predicted probability distributions.
        """
        return (self.image_weight * image_probs) + (self.sensor_weight * sensor_probs)
