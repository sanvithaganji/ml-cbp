"""
Dataset Validation Module.
Verifies file existence, label consistency, sensor physical bounds,
missing values, duplicate rows, and image-to-sensor pairing.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BOUNDS = {
    "ambient_temperature": (5.0, 50.0),
    "soil_temperature": (5.0, 45.0),
    "humidity": (10.0, 100.0),
    "soil_moisture": (0.0, 100.0),
    "leaf_wetness": (0.0, 24.0),
}

VALID_CLASSES = {"Healthy", "Anthracnose", "Leaf Spot"}


def validate_dataset(
    metadata_df: pd.DataFrame,
    base_image_dir: str,
    feature_names: Optional[List[str]] = None,
    bounds: Optional[Dict[str, tuple]] = None,
    quick_image_check_limit: Optional[int] = 500,
) -> Dict[str, Any]:
    """
    Perform rigorous validation of metadata, image paths, and environmental sensors.
    
    Args:
        metadata_df: DataFrame containing image paths, labels, and sensor values.
        base_image_dir: Base directory to resolve relative image paths.
        feature_names: List of environmental feature columns.
        bounds: Physical bounds dictionary (min, max).
        quick_image_check_limit: Max images to verify opening for speed (None for all).
        
    Returns:
        Validation report dictionary with findings and error flags.
    """
    if feature_names is None:
        feature_names = list(DEFAULT_BOUNDS.keys())
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    report: Dict[str, Any] = {
        "is_valid": True,
        "total_records": len(metadata_df),
        "issues": [],
        "warnings": [],
        "summary": {},
    }

    if metadata_df.empty:
        report["is_valid"] = False
        report["issues"].append("Metadata DataFrame is empty.")
        return report

    # 1. Required column validation
    required_cols = ["image_path", "label"] + feature_names
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["issues"].append(f"Missing required columns: {missing_cols}")
        return report

    # 2. Duplicate checks
    dup_paths = metadata_df["image_path"].duplicated().sum()
    if dup_paths > 0:
        report["warnings"].append(f"Found {dup_paths} duplicate image paths in metadata.")

    # 3. Label validation and distribution
    unique_labels = set(metadata_df["label"].dropna().unique())
    invalid_labels = unique_labels - VALID_CLASSES
    if invalid_labels:
        report["is_valid"] = False
        report["issues"].append(f"Found invalid labels: {list(invalid_labels)}. Allowed: {list(VALID_CLASSES)}")

    class_counts = metadata_df["label"].value_counts().to_dict()
    report["summary"]["class_distribution"] = class_counts
    total_samples = len(metadata_df)
    report["summary"]["class_proportions"] = {
        cls: round(count / total_samples, 4) for cls, count in class_counts.items()
    }

    # 4. Sensor readings: missing values and physical boundaries
    sensor_stats = {}
    total_missing_sensors = 0
    total_out_of_bounds = 0

    for feat in feature_names:
        series = metadata_df[feat]
        missing_count = int(series.isna().sum())
        total_missing_sensors += missing_count

        valid_vals = series.dropna()
        if len(valid_vals) > 0:
            min_val = float(valid_vals.min())
            max_val = float(valid_vals.max())
            mean_val = float(valid_vals.mean())
            std_val = float(valid_vals.std()) if len(valid_vals) > 1 else 0.0

            b_min, b_max = bounds.get(feat, (-np.inf, np.inf))
            out_bounds_mask = (valid_vals < b_min) | (valid_vals > b_max)
            oob_count = int(out_bounds_mask.sum())
            total_out_of_bounds += oob_count

            sensor_stats[feat] = {
                "missing": missing_count,
                "out_of_bounds": oob_count,
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "expected_range": [b_min, b_max],
            }
        else:
            sensor_stats[feat] = {
                "missing": missing_count,
                "out_of_bounds": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "expected_range": list(bounds.get(feat, (-np.inf, np.inf))),
            }

    report["summary"]["sensor_stats"] = sensor_stats
    if total_missing_sensors > 0:
        report["warnings"].append(
            f"Found {total_missing_sensors} missing sensor values across features. Imputation will be needed."
        )
    if total_out_of_bounds > 0:
        report["warnings"].append(
            f"Found {total_out_of_bounds} sensor values exceeding biological/physical bounds."
        )

    # 5. Image file verification
    missing_images = []
    corrupt_images = []
    
    check_subset = metadata_df
    if quick_image_check_limit and len(metadata_df) > quick_image_check_limit:
        check_subset = metadata_df.sample(n=quick_image_check_limit, random_state=42)
        report["warnings"].append(
            f"Verified {quick_image_check_limit} of {len(metadata_df)} images for existence and file integrity."
        )

    for _, row in check_subset.iterrows():
        rel_path = str(row["image_path"])
        full_path = os.path.join(base_image_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        
        if not os.path.exists(full_path):
            missing_images.append(rel_path)
        else:
            try:
                with Image.open(full_path) as img:
                    img.verify()
            except Exception:
                corrupt_images.append(rel_path)

    if missing_images:
        report["is_valid"] = False
        report["issues"].append(
            f"Failed to find {len(missing_images)} image files on disk. First 3: {missing_images[:3]}"
        )
    if corrupt_images:
        report["is_valid"] = False
        report["issues"].append(
            f"Detected {len(corrupt_images)} corrupted image files. First 3: {corrupt_images[:3]}"
        )

    # 6. Grouping and metadata pairing documentation
    has_plant_id = "plant_id" in metadata_df.columns
    has_timestamp = "timestamp" in metadata_df.columns
    has_batch_id = "batch_id" in metadata_df.columns

    report["summary"]["pairing_metadata"] = {
        "has_plant_id": has_plant_id,
        "has_timestamp": has_timestamp,
        "has_batch_id": has_batch_id,
        "grouping_column_available": has_plant_id or has_batch_id,
        "granularity": "plant-level / timestamped" if (has_plant_id or has_timestamp) else "individual-record",
    }

    if not has_plant_id:
        report["warnings"].append(
            "No 'plant_id' column found. Group-based train/test splitting cannot guarantee zero plant leakage."
        )

    return report
