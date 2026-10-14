"""
Data Preprocessing Module.
Handles image transforms (training augmentation, validation resizing)
and environmental sensor normalization (fit exclusively on the training split).
"""

import os
import logging
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torchvision.transforms as T

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transforms(
    input_size: int = 224,
    is_training: bool = False
) -> T.Compose:
    """
    Build torchvision transformation pipeline.
    
    Args:
        input_size: Image dimension (224 for EfficientNet-B0)
        is_training: If True, apply data augmentation
        
    Returns:
        torchvision.transforms.Compose pipeline
    """
    if is_training:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class EnvironmentalPreprocessor:
    """
    Fits and applies median imputation and standard scaling to environmental sensors.
    Strictly fit ONLY on the training split to prevent data leakage.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "EnvironmentalPreprocessor":
        """
        Fit imputer and scaler on training split DataFrame.
        """
        X = train_df[self.feature_names].to_numpy(dtype=np.float32)
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        logger.info("Fitted environmental preprocessor on %d training samples.", len(train_df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform given DataFrame using fitted parameters.
        Preserves strict feature ordering.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")
        
        # Ensure correct column ordering
        X = df[self.feature_names].to_numpy(dtype=np.float32)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled.astype(np.float32)

    def transform_single(self, sensor_dict: Dict[str, float]) -> np.ndarray:
        """
        Transform a single environmental measurement dictionary.
        Returns 1D array of shape (num_features,).
        """
        row = [sensor_dict.get(feat, np.nan) for feat in self.feature_names]
        df_single = pd.DataFrame([row], columns=self.feature_names)
        transformed = self.transform(df_single)
        return transformed[0]

    def save(self, file_path: str) -> None:
        """Serialize preprocessor state to disk using joblib."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        payload = {
            "feature_names": self.feature_names,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, file_path)
        logger.info("Saved environmental preprocessor to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "EnvironmentalPreprocessor":
        """Load preprocessor state from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        payload = joblib.load(file_path)
        instance = cls(feature_names=payload["feature_names"])
        instance.imputer = payload["imputer"]
        instance.scaler = payload["scaler"]
        instance.is_fitted = payload["is_fitted"]
        logger.info("Loaded environmental preprocessor from %s", file_path)
        return instance
