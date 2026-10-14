"""
Image-Only Disease Classification Model using EfficientNet-B0.
Extracts 1280-dimensional visual feature vectors and classifies into 3 classes.
"""

from typing import Tuple, Optional
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class LeafDiseaseEfficientNet(nn.Module):
    """
    Image-only classification network based on pretrained EfficientNet-B0.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # EfficientNet-B0 feature extractor and pooling
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = 1280

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled feature vector of shape (batch_size, 1280).
        """
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)
        return flat

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Image tensor of shape (batch_size, 3, 224, 224)
            return_features: If True, returns (logits, features)
        Returns:
            logits: (batch_size, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
