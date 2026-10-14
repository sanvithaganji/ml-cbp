"""
Training Pipeline for End-to-End Multimodal Fusion Model.
Combines leaf image features from EfficientNet-B0 and environmental sensor features.
Employs differential learning rates for pretrained backbone vs. fusion heads.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.reproducibility import set_seed, get_device
from src.utils.metrics import compute_classification_metrics
from src.data.preprocessing import EnvironmentalPreprocessor
from src.data.dataset import split_dataset, JamunMultimodalDataset, CLASS_NAMES, CLASS_TO_IDX, FEATURE_NAMES
from src.models.fusion_model import MultimodalFusionNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_fusion_model(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("random_seed", 42)
    set_seed(seed)
    device = get_device(config.get("training", {}).get("device", "auto"))
    logger.info("Training Multimodal Fusion model using device: %s", device)

    metadata_path = config.get("paths", {}).get("metadata_file", "data/metadata.csv")
    data_dir = config.get("paths", {}).get("data_dir", "data")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    preproc_dir = config.get("paths", {}).get("preprocessor_dir", "models/preprocessors")
    metrics_dir = config.get("paths", {}).get("metrics_dir", "outputs/metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
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
    preproc_path = os.path.join(preproc_dir, "sensor_scaler.joblib")
    if os.path.exists(preproc_path):
        preprocessor = EnvironmentalPreprocessor.load(preproc_path)
    else:
        logger.info("Preprocessor not found. Fitting on train split now...")
        preprocessor = EnvironmentalPreprocessor(feature_names=feat_names).fit(train_df)
        preprocessor.save(preproc_path)

    batch_size = config["training"].get("batch_size", 16)
    fusion_cfg = config["training"]["multimodal_fusion"]

    train_ds = JamunMultimodalDataset(
        train_df,
        base_image_dir=data_dir,
        preprocessor=preprocessor,
        feature_names=feat_names,
        is_training=True,
    )
    val_ds = JamunMultimodalDataset(
        val_df,
        base_image_dir=data_dir,
        preprocessor=preprocessor,
        feature_names=feat_names,
        is_training=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultimodalFusionNet(
        num_classes=len(CLASS_NAMES),
        num_sensor_features=len(feat_names),
        pretrained=fusion_cfg.get("pretrained", True),
        image_embed_dim=fusion_cfg.get("image_embed_dim", 128),
        sensor_embed_dim=fusion_cfg.get("sensor_embed_dim", 32),
        fusion_hidden_dim=fusion_cfg.get("fusion_hidden_dim", 64),
        dropout=float(fusion_cfg.get("dropout", 0.3)),
    ).to(device)

    # Differential learning rate: slow for pretrained backbone, faster for fusion heads
    backbone_params = list(model.visual_backbone.parameters())
    head_params = (
        list(model.visual_projector.parameters())
        + list(model.sensor_branch.parameters())
        + list(model.fusion_head.parameters())
    )

    backbone_lr = float(fusion_cfg.get("backbone_lr", 5e-5))
    head_lr = float(fusion_cfg.get("head_lr", 5e-4))
    weight_decay = float(fusion_cfg.get("weight_decay", 1e-4))

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Class weighting
    counts = [np.sum(train_df["label"] == c) for c in CLASS_NAMES]
    total_counts = len(train_df)
    weights = [total_counts / (len(CLASS_NAMES) * max(c, 1)) for c in counts]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    epochs = fusion_cfg.get("epochs", 15)
    patience = fusion_cfg.get("early_stopping_patience", 4)
    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}

    best_checkpoint_path = os.path.join(checkpoints_dir, "multimodal_fusion_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            sensors = batch["sensor"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images, sensors)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                sensors = batch["sensor"].to(device)
                labels = batch["label"].to(device)

                logits = model(images, sensors)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.append(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_ds)
        all_probs = np.vstack(all_probs)
        val_metrics = compute_classification_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=all_probs,
            class_names=CLASS_NAMES,
        )

        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]
        scheduler.step(val_f1)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_acc"].append(round(val_acc, 4))

        logger.info(
            "Fusion Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro F1: %.4f",
            epoch, epochs, epoch_train_loss, epoch_val_loss, val_acc, val_f1
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "classes": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "MultimodalFusionNet",
                "hyperparameters": fusion_cfg,
            }, best_checkpoint_path)
            logger.info("  --> Saved new best Multimodal Fusion checkpoint (Val Macro F1: %.4f)", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered for Multimodal Fusion at epoch %d.", epoch)
                break

    history_path = os.path.join(metrics_dir, "history_fusion.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Multimodal fusion training complete. Best Val Macro F1: %.4f", best_val_f1)
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train_fusion_model(args.config)
