"""
Dataset Inspection Script.
Conducts full validation of metadata, image files, sensor distributions,
missing/out-of-bounds readings, and record pairing.
Outputs a structured JSON inspection report to outputs/dataset_inspection_report.json.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import yaml

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.validation import validate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inspect and validate dataset integrity.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        logger.error("Metadata file not found at %s. Please generate demo data or add real dataset.", metadata_path)
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    logger.info("Loaded metadata containing %d rows from %s", len(df), metadata_path)

    # Convert bounds from config
    raw_bounds = config.get("environmental_features", {}).get("bounds", {})
    bounds = {k: tuple(v) for k, v in raw_bounds.items()}
    feat_names = config.get("environmental_features", {}).get("names", [])

    report = validate_dataset(
        metadata_df=df,
        base_image_dir=args.data_dir,
        feature_names=feat_names,
        bounds=bounds,
        quick_image_check_limit=None,  # Check all images
    )

    outputs_dir = config.get("paths", {}).get("outputs_dir", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    report_path = os.path.join(outputs_dir, "dataset_inspection_report.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("================ DATASET INSPECTION SUMMARY ================")
    logger.info("Validation Passed: %s", report["is_valid"])
    logger.info("Total Records: %d", report["total_records"])
    logger.info("Class Distribution: %s", report["summary"].get("class_distribution", {}))
    logger.info("Grouping Granularity: %s", report["summary"].get("pairing_metadata", {}).get("granularity"))
    
    if report["warnings"]:
        logger.info("Warnings (%d):", len(report["warnings"]))
        for w in report["warnings"]:
            logger.info("  - %s", w)
            
    if report["issues"]:
        logger.error("Issues (%d):", len(report["issues"]))
        for iss in report["issues"]:
            logger.error("  - %s", iss)

    logger.info("Full JSON report saved to: %s", report_path)


if __name__ == "__main__":
    main()
