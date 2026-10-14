"""
Training Pipeline for Environmental Sensor Baselines.
1. Fits EnvironmentalPreprocessor strictly on the training split.
2. Trains Random Forest classifier (scikit-learn).
3. Trains SensorMLP classifier (PyTorch).
Saves fitted preprocessor and checkpoints.
"""

import os
import sys
import json
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.reproducibility import set_seed, get_device
from src.utils.metrics import compute_classification_metrics
from src.data.preprocessing import EnvironmentalPreprocessor
from src.data.dataset import split_dataset, CLASS_NAMES, CLASS_TO_IDX, FEATURE_NAMES
from src.models.sensor_models import SensorMLP, SensorRandomForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_sensor_models(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("random_seed", 42)
    set_seed(seed)
    device = get_device(config.get("training", {}).get("device", "auto"))

    metadata_path = config.get("paths", {}).get("metadata_file", "data/metadata.csv")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    preproc_dir = config.get("paths", {}).get("preprocessor_dir", "models/preprocessors")
    metrics_dir = config.get("paths", {}).get("metrics_dir", "outputs/metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(preproc_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=config["data_split"]["train_ratio"],
        val_ratio=config["data_split"]["val_ratio"],
        test_ratio=config["data_split"]["test_ratio"],
        seed=seed,
        group_col="plant_id" if config["data_split"].get("group_by_plant") else None,
    )

    feat_names = config.get("environmental_features", {}).get("names", FEATURE_NAMES)

    # 1. Fit preprocessor STRICTLY on train_df
    preprocessor = EnvironmentalPreprocessor(feature_names=feat_names)
    preprocessor.fit(train_df)
    preproc_path = os.path.join(preproc_dir, "sensor_scaler.joblib")
    preprocessor.save(preproc_path)

    X_train = preprocessor.transform(train_df)
    y_train = np.array([CLASS_TO_IDX[lbl] for lbl in train_df["label"]], dtype=np.int64)

    X_val = preprocessor.transform(val_df)
    y_val = np.array([CLASS_TO_IDX[lbl] for lbl in val_df["label"]], dtype=np.int64)

    # =========================================================================
    # Model 1: Random Forest Classifier
    # =========================================================================
    logger.info("--- Training Random Forest Sensor Baseline ---")
    rf_cfg = config["training"].get("sensor_rf", {})
    rf_model = SensorRandomForest(
        n_estimators=rf_cfg.get("n_estimators", 120),
        max_depth=rf_cfg.get("max_depth", 8),
        random_state=seed,
        class_weight=rf_cfg.get("class_weight", "balanced"),
    )
    rf_model.fit(X_train, y_train)

    rf_val_preds = rf_model.predict(X_val)
    rf_val_probs = rf_model.predict_proba(X_val)
    rf_metrics = compute_classification_metrics(y_val, rf_val_preds, rf_val_probs, class_names=CLASS_NAMES)
    logger.info("Random Forest Val Acc: %.4f | Val Macro F1: %.4f", rf_metrics["accuracy"], rf_metrics["macro_f1"])

    rf_save_path = os.path.join(checkpoints_dir, "sensor_rf_best.joblib")
    rf_model.save(rf_save_path)

    # =========================================================================
    # Model 2: Sensor MLP (PyTorch)
    # =========================================================================
    logger.info("--- Training Sensor MLP Baseline (PyTorch) ---")
    mlp_cfg = config["training"].get("sensor_mlp", {})
    hidden_dims = tuple(mlp_cfg.get("hidden_dims", [64, 32]))

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_ds, batch_size=config["training"].get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"].get("batch_size", 16), shuffle=False)

    mlp_model = SensorMLP(
        input_dim=len(feat_names),
        hidden_dims=hidden_dims,
        num_classes=len(CLASS_NAMES),
        dropout=float(mlp_cfg.get("dropout", 0.2)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        mlp_model.parameters(),
        lr=float(mlp_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )

    epochs = mlp_cfg.get("epochs", 30)
    patience = mlp_cfg.get("early_stopping_patience", 6)
    best_val_f1 = -1.0
    patience_counter = 0
    mlp_history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    mlp_checkpoint_path = os.path.join(checkpoints_dir, "sensor_mlp_best.pt")

    for epoch in range(1, epochs + 1):
        mlp_model.train()
        running_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = mlp_model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_b.size(0)

        epoch_train_loss = running_loss / len(train_ds)

        mlp_model.eval()
        val_loss = 0.0
        preds_list, labels_list, probs_list = [], [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = mlp_model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * x_b.size(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
                preds_list.extend(np.argmax(probs, axis=1))
                labels_list.extend(y_b.cpu().numpy())

        epoch_val_loss = val_loss / len(val_ds)
        probs_mat = np.vstack(probs_list)
        val_m = compute_classification_metrics(np.array(labels_list), np.array(preds_list), probs_mat, CLASS_NAMES)
        val_f1 = val_m["macro_f1"]

        mlp_history["train_loss"].append(round(epoch_train_loss, 4))
        mlp_history["val_loss"].append(round(epoch_val_loss, 4))
        mlp_history["val_macro_f1"].append(round(val_f1, 4))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": mlp_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_m,
                "classes": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "SensorMLP",
                "hidden_dims": hidden_dims,
                "feature_names": feat_names,
            }, mlp_checkpoint_path)
            logger.info("  --> Saved new best SensorMLP checkpoint (Val Macro F1: %.4f)", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered for SensorMLP at epoch %d.", epoch)
                break

    mlp_history_path = os.path.join(metrics_dir, "history_sensor_mlp.json")
    with open(mlp_history_path, "w") as f:
        json.dump(mlp_history, f, indent=2)

    logger.info("Sensor models training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train_sensor_models(args.config)
