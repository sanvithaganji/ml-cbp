"""
Unit Tests for Data Validation, Preprocessing, and Dataset Loading.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.validation import validate_dataset, DEFAULT_BOUNDS
from src.data.preprocessing import EnvironmentalPreprocessor, get_image_transforms
from src.data.dataset import split_dataset, JamunMultimodalDataset, CLASS_NAMES, FEATURE_NAMES


@pytest.fixture
def sample_metadata():
    # 30 synthetic samples across 3 plants
    data = []
    classes = ["Healthy", "Anthracnose", "Leaf Spot"]
    for i in range(30):
        data.append({
            "image_path": f"dummy_{i}.jpg",
            "label": classes[i % 3],
            "ambient_temperature": 25.0 + (i % 5),
            "soil_temperature": 23.0 + (i % 4),
            "humidity": 65.0 + (i % 20),
            "soil_moisture": 45.0 + (i % 15),
            "leaf_wetness": float(i % 10),
            "plant_id": f"PLANT_{i % 10:02d}",
        })
    return pd.DataFrame(data)


def test_validation_logic(sample_metadata, tmp_path):
    report = validate_dataset(
        metadata_df=sample_metadata,
        base_image_dir=str(tmp_path),
        feature_names=FEATURE_NAMES,
        bounds=DEFAULT_BOUNDS,
        quick_image_check_limit=None,
    )
    # File check will warn/fail for dummy non-existent files, but schema and distribution pass
    assert "class_distribution" in report["summary"]
    assert report["summary"]["class_distribution"]["Healthy"] == 10
    assert report["summary"]["pairing_metadata"]["grouping_column_available"] is True


def test_split_prevents_leakage(sample_metadata):
    train_df, val_df, test_df = split_dataset(
        sample_metadata,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        group_col="plant_id",
    )
    # Ensure no plant_id overlaps between train and test
    train_plants = set(train_df["plant_id"])
    test_plants = set(test_df["plant_id"])
    assert len(train_plants.intersection(test_plants)) == 0


def test_environmental_preprocessor(sample_metadata, tmp_path):
    preproc = EnvironmentalPreprocessor(feature_names=FEATURE_NAMES)
    preproc.fit(sample_metadata)
    assert preproc.is_fitted is True

    scaled = preproc.transform(sample_metadata)
    assert scaled.shape == (30, 5)
    # Mean of standardized columns should be close to 0
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)

    # Test serialization
    save_file = str(tmp_path / "scaler.joblib")
    preproc.save(save_file)
    loaded = EnvironmentalPreprocessor.load(save_file)
    assert loaded.is_fitted is True
    assert loaded.feature_names == FEATURE_NAMES


def test_multimodal_dataset_fallback(sample_metadata, tmp_path):
    # Tests that missing images gracefully return fallback image tensor
    preproc = EnvironmentalPreprocessor(feature_names=FEATURE_NAMES).fit(sample_metadata)
    ds = JamunMultimodalDataset(
        sample_metadata,
        base_image_dir=str(tmp_path),
        preprocessor=preproc,
        is_training=False,
    )
    assert len(ds) == 30
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["sensor"].shape == (5,)
    assert isinstance(item["label"].item(), int)
