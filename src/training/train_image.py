"""
Training Pipeline for Image-Only Baseline (EfficientNet-B0).
Includes data augmentation, AdamW optimization, class weighting,
learning rate scheduling, early stopping, and checkpointing.
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
from src.data.dataset import split_dataset, JamunMultimodalDataset, CLASS_NAMES, CLASS_TO_IDX
from src.models.image_model import LeafDiseaseEfficientNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_image_model(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("random_seed", 42)
    set_seed(seed)
    device = get_device(config.get("training", {}).get("device", "auto"))
    logger.info("Training Image-Only model using device: %s", device)

    metadata_path = config.get("paths", {}).get("metadata_file", "data/metadata.csv")
    data_dir = config.get("paths", {}).get("data_dir", "data")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
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

    batch_size = config["training"].get("batch_size", 16)
    img_cfg = config["training"]["image_model"]

    train_ds = JamunMultimodalDataset(
        train_df,
        base_image_dir=data_dir,
        is_training=True,
        input_size=img_cfg.get("input_size", 224),
    )
    val_ds = JamunMultimodalDataset(
        val_df,
        base_image_dir=data_dir,
        is_training=False,
        input_size=img_cfg.get("input_size", 224),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class weighting to handle imbalance
    if img_cfg.get("use_class_weights", True):
        counts = [np.sum(train_df["label"] == c) for c in CLASS_NAMES]
        total_counts = len(train_df)
        weights = [total_counts / (len(CLASS_NAMES) * max(c, 1)) for c in counts]
        class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        logger.info("Computed class weights: %s", weights)
    else:
        criterion = nn.CrossEntropyLoss()

    model = LeafDiseaseEfficientNet(
        num_classes=len(CLASS_NAMES),
        pretrained=img_cfg.get("pretrained", True),
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(img_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(img_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs = img_cfg.get("epochs", 15)
    patience = img_cfg.get("early_stopping_patience", 4)
    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}

    best_checkpoint_path = os.path.join(checkpoints_dir, "image_efficientnet_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
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
                labels = batch["label"].to(device)
                logits = model(images)
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
            "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro F1: %.4f",
            epoch, epochs, epoch_train_loss, epoch_val_loss, val_acc, val_f1
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "classes": CLASS_NAMES,
                "class_to_idx": CLASS_TO_IDX,
                "architecture": "LeafDiseaseEfficientNet",
                "backbone": "efficientnet_b0",
            }, best_checkpoint_path)
            logger.info("  --> Saved new best image model checkpoint (Val Macro F1: %.4f)", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
                break

    # Save history
    history_path = os.path.join(metrics_dir, "history_image.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Image-only model training complete. Best Val Macro F1: %.4f", best_val_f1)
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train_image_model(args.config)
