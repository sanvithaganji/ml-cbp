"""
Sensor-Only Disease Classification Models.
Implements:
1. SensorMLP: PyTorch Multi-Layer Perceptron with compact embedding support.
2. SensorRandomForest: Scikit-learn Random Forest baseline.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class SensorMLP(nn.Module):
    """
    PyTorch MLP for classifying crop disease from 5 environmental sensors.
    Projects 5-dim input into 32-dim embedding before the final linear layer.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Tuple[int, int] = (64, 32),
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = hidden_dims[1]
        self.num_classes = num_classes

        # Embedding branch
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Output classification head
        self.classifier = nn.Linear(hidden_dims[1], num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract compact sensor embedding of shape (batch_size, 32).
        """
        # If batch size is 1 during evaluation or inference, handle BatchNorm
        if x.size(0) == 1 and self.training:
            return self.embedding_net(x)
        return self.embedding_net(x)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Sensor tensor of shape (batch_size, 5)
            return_embedding: If True, returns (logits, embedding)
        """
        embedding = self.embedding_net(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class SensorRandomForest:
    """
    Random Forest Classifier wrapper for tabular environmental features.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 8,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SensorRandomForest":
        """Fit model on tabular features X and targets y."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distributions."""
        return self.model.predict_proba(X)

    def save(self, file_path: str) -> None:
        """Serialize Random Forest model to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.model, file_path)
        logger.info("Saved Random Forest model to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "SensorRandomForest":
        """Load Random Forest model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        instance = cls()
        instance.model = joblib.load(file_path)
        instance.is_fitted = True
        logger.info("Loaded Random Forest model from %s", file_path)
        return instance
