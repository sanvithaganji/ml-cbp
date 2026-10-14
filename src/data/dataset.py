"""
Dataset Loader and PyTorch Dataset Implementation.
Implements reproducible, group-aware splitting and multimodal data loading.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit

from src.data.preprocessing import get_image_transforms, EnvironmentalPreprocessor

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Healthy", "Anthracnose", "Leaf Spot"]
CLASS_TO_IDX = {"Healthy": 0, "Anthracnose": 1, "Leaf Spot": 2}
FEATURE_NAMES = [
    "ambient_temperature",
    "soil_temperature",
    "humidity",
    "soil_moisture",
    "leaf_wetness",
]


def split_dataset(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_col: Optional[str] = "plant_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata DataFrame into train, val, and test subsets.
    If `group_col` is present in dataframe, uses GroupShuffleSplit to prevent data leakage.
    Otherwise uses StratifiedShuffleSplit on class labels.
    """
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-4, f"Split ratios must sum to 1.0, got {total}"

    df = metadata_df.copy().reset_index(drop=True)

    if group_col and group_col in df.columns and df[group_col].nunique() >= 3:
        logger.info("Using Group-based split on column '%s' to eliminate cross-plant leakage.", group_col)
        # First split off test set
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))
        
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Then split train and val
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        logger.info("Using Stratified split on 'label' (no grouping column present).")
        # Split off test set
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(sss_test.split(df, df["label"]))

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split train and val
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
        train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df["label"]))

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Split summary: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_df), len(train_df)/len(df)*100,
        len(val_df), len(val_df)/len(df)*100,
        len(test_df), len(test_df)/len(df)*100,
    )
    return train_df, val_df, test_df


class JamunMultimodalDataset(Dataset):
    """
    Multimodal Dataset for Jamun Plant Disease Diagnosis.
    Returns:
        - image: Tensor of shape (3, 224, 224)
        - sensor_vector: Float tensor of shape (5,) with normalized values
        - label: Long tensor with class index (0: Healthy, 1: Anthracnose, 2: Leaf Spot)
        - metadata: Dictionary containing raw sensor values, image path, etc.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_image_dir: str,
        preprocessor: Optional[EnvironmentalPreprocessor] = None,
        feature_names: Optional[List[str]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        is_training: bool = False,
        input_size: int = 224,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.base_image_dir = base_image_dir
        self.feature_names = feature_names or FEATURE_NAMES
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.is_training = is_training
        self.input_size = input_size
        self.transforms = get_image_transforms(input_size=input_size, is_training=is_training)

        # Precompute preprocessed sensor array if preprocessor is provided
        if preprocessor is not None:
            self.sensor_data = preprocessor.transform(self.df)
        else:
            # Fallback: fillna with 0.0
            self.sensor_data = self.df[self.feature_names].fillna(0.0).to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_img_path = str(row["image_path"])
        full_path = os.path.join(self.base_image_dir, rel_img_path) if not os.path.isabs(rel_img_path) else rel_img_path

        # Load RGB image
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s: %s. Using blank fallback.", full_path, str(e))
            image = Image.new("RGB", (self.input_size, self.input_size), color=(128, 128, 128))

        image_tensor = self.transforms(image)

        # Environmental sensors
        sensor_vector = torch.from_numpy(np.array(self.sensor_data[idx], copy=True)).float()

        # Target label
        label_str = str(row["label"])
        label_idx = self.class_to_idx.get(label_str, 0)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        # Raw metadata dict
        raw_sensors = {feat: float(row.get(feat, np.nan)) for feat in self.feature_names}

        return {
            "image": image_tensor,
            "sensor": sensor_vector,
            "label": label_tensor,
            "image_path": rel_img_path,
            "raw_sensors": raw_sensors,
            "label_name": label_str,
        }
