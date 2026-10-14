"""
Comprehensive Evaluation and Fair Comparison Module.
Evaluates all trained baselines and multimodal architectures on the held-out test split.
Computes metrics, latency benchmarks, generates comparison plots, and conducts ablation analysis.
"""

import os
import sys
import json
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.reproducibility import set_seed, get_device
from src.utils.metrics import compute_classification_metrics, measure_inference_latency
from src.data.preprocessing import EnvironmentalPreprocessor
from src.data.dataset import split_dataset, JamunMultimodalDataset, CLASS_NAMES, CLASS_TO_IDX, FEATURE_NAMES
from src.models.image_model import LeafDiseaseEfficientNet
from src.models.sensor_models import SensorMLP, SensorRandomForest
from src.models.fusion_model import MultimodalFusionNet, LateFusionClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_metric_comparisons(results: dict, output_path: str):
    """Generate side-by-side bar chart of model evaluation metrics."""
    models = list(results.keys())
    metrics_to_plot = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]

    for i, m_key in enumerate(metrics_to_plot):
        vals = [results[m][m_key] for m in models]
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=metric_labels[i], color=palette[i], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Model Performance Comparison on Held-Out Test Set", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(0.90, color="crimson", linestyle="--", linewidth=1.2, label="Aspirational Benchmark (90%)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved metrics comparison plot to %s", output_path)


def plot_confusion_matrices(results: dict, output_path: str):
    """Plot confusion matrices for all evaluated models in a 2x3 grid."""
    models = list(results.keys())
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), dpi=300)
    axes = np.array(axes).flatten()

    for idx, m_name in enumerate(models):
        ax = axes[idx]
        cm = np.array(results[m_name]["confusion_matrix"])
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{m_name}\n(F1: {results[m_name]['macro_f1']:.3f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=9)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Normalized Confusion Matrices on Test Population", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrices plot to %s", output_path)


def plot_roc_curves(prob_dict: dict, y_true: np.ndarray, output_path: str):
    """Plot One-vs-Rest ROC curves for each model across all classes."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 5), dpi=300)

    for c_idx, c_name in enumerate(CLASS_NAMES):
        ax = axes[c_idx]
        y_binary = (y_true == c_idx).astype(int)

        for m_name, probs in prob_dict.items():
            if probs is not None and probs.shape[1] == len(CLASS_NAMES):
                fpr, tpr, _ = roc_curve(y_binary, probs[:, c_idx])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{m_name} (AUC={roc_auc:.3f})", linewidth=1.8)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
        ax.set_title(f"ROC: {c_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("One-vs-Rest ROC Curves across Target Classes", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curves plot to %s", output_path)


def plot_loss_curves(metrics_dir: str, output_path: str):
    """Plot training and validation loss progression from saved histories."""
    hist_files = {
        "Image Baseline": os.path.join(metrics_dir, "history_image.json"),
        "Sensor MLP": os.path.join(metrics_dir, "history_sensor_mlp.json"),
        "Multimodal Fusion": os.path.join(metrics_dir, "history_fusion.json"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    has_data = False

    for label, hpath in hist_files.items():
        if os.path.exists(hpath):
            with open(hpath, "r") as f:
                hist = json.load(f)
            epochs = range(1, len(hist.get("train_loss", [])) + 1)
            if len(epochs) > 0:
                has_data = True
                axes[0].plot(epochs, hist["train_loss"], label=f"{label} (Train)", linestyle="--", linewidth=1.5)
                axes[0].plot(epochs, hist["val_loss"], label=f"{label} (Val)", linewidth=2.0)
                if "val_macro_f1" in hist and len(hist["val_macro_f1"]) == len(epochs):
                    axes[1].plot(epochs, hist["val_macro_f1"], label=f"{label}", linewidth=2.0)

    if has_data:
        axes[0].set_title("Training & Validation Loss Progression", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=9)
        axes[0].set_ylabel("Cross-Entropy Loss", fontsize=9)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Validation Macro F1 Score Progression", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=9)
        axes[1].set_ylabel("Macro F1", fontsize=9)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info("Saved loss progression curves to %s", output_path)


def run_evaluation(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("project", {}).get("random_seed", 42)
    set_seed(seed)
    device = get_device(config.get("training", {}).get("device", "auto"))
    logger.info("Running evaluation on device: %s", device)

    metadata_path = config.get("paths", {}).get("metadata_file", "data/metadata.csv")
    data_dir = config.get("paths", {}).get("data_dir", "data")
    checkpoints_dir = config.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    preproc_dir = config.get("paths", {}).get("preprocessor_dir", "models/preprocessors")
    metrics_dir = config.get("paths", {}).get("metrics_dir", "outputs/metrics")
    plots_dir = config.get("paths", {}).get("plots_dir", "outputs/plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(metadata_path)
    _, _, test_df = split_dataset(
        df,
        train_ratio=config["data_split"]["train_ratio"],
        val_ratio=config["data_split"]["val_ratio"],
        test_ratio=config["data_split"]["test_ratio"],
        seed=seed,
        group_col="plant_id" if config["data_split"].get("group_by_plant") else None,
    )

    feat_names = config.get("environmental_features", {}).get("names", FEATURE_NAMES)
    preproc_path = os.path.join(preproc_dir, "sensor_scaler.joblib")
    preprocessor = EnvironmentalPreprocessor.load(preproc_path)

    test_ds = JamunMultimodalDataset(
        test_df,
        base_image_dir=data_dir,
        preprocessor=preprocessor,
        feature_names=feat_names,
        is_training=False,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"]], dtype=np.int64)
    results = {}
    probs_dict = {}

    # Sample single input for measuring latency
    first_item = test_ds[0]
    sample_img = first_item["image"].unsqueeze(0).to(device)
    sample_sensor = first_item["sensor"].unsqueeze(0).to(device)

    # =========================================================================
    # 1. Image-Only EfficientNet-B0
    # =========================================================================
    img_ckpt_path = os.path.join(checkpoints_dir, "image_efficientnet_best.pt")
    if os.path.exists(img_ckpt_path):
        logger.info("Evaluating Image-Only baseline...")
        img_model = LeafDiseaseEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
        ckpt = torch.load(img_ckpt_path, map_location=device)
        img_model.load_state_dict(ckpt["model_state_dict"])
        img_model.eval()

        img_probs = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)
                logits = img_model(imgs)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                img_probs.append(p)
        img_probs = np.vstack(img_probs)
        img_preds = np.argmax(img_probs, axis=1)

        def img_pred_fn(x):
            with torch.no_grad():
                return img_model(x)
        lat = measure_inference_latency(img_pred_fn, (sample_img,))

        m = compute_classification_metrics(y_true, img_preds, img_probs, CLASS_NAMES)
        m["latency_ms_per_sample"] = lat
        results["Image-Only (EfficientNet-B0)"] = m
        probs_dict["Image-Only (EfficientNet-B0)"] = img_probs
    else:
        logger.warning("Image checkpoint not found at %s. Skipping.", img_ckpt_path)

    # =========================================================================
    # 2. Sensor-Only Random Forest
    # =========================================================================
    rf_path = os.path.join(checkpoints_dir, "sensor_rf_best.joblib")
    if os.path.exists(rf_path):
        logger.info("Evaluating Sensor-Only Random Forest...")
        rf_model = SensorRandomForest.load(rf_path)
        X_test = preprocessor.transform(test_df)
        rf_probs = rf_model.predict_proba(X_test)
        rf_preds = rf_model.predict(X_test)

        sample_x_np = X_test[:1]
        lat = measure_inference_latency(rf_model.predict_proba, (sample_x_np,))

        m = compute_classification_metrics(y_true, rf_preds, rf_probs, CLASS_NAMES)
        m["latency_ms_per_sample"] = lat
        results["Sensor-Only (Random Forest)"] = m
        probs_dict["Sensor-Only (Random Forest)"] = rf_probs
    else:
        logger.warning("Sensor RF model not found at %s. Skipping.", rf_path)

    # =========================================================================
    # 3. Sensor-Only MLP (PyTorch)
    # =========================================================================
    mlp_ckpt_path = os.path.join(checkpoints_dir, "sensor_mlp_best.pt")
    if os.path.exists(mlp_ckpt_path):
        logger.info("Evaluating Sensor-Only MLP...")
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device)
        mlp_model = SensorMLP(
            input_dim=len(feat_names),
            hidden_dims=tuple(mlp_ckpt.get("hidden_dims", (64, 32))),
            num_classes=len(CLASS_NAMES),
        ).to(device)
        mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
        mlp_model.eval()

        mlp_probs = []
        with torch.no_grad():
            for batch in test_loader:
                sens = batch["sensor"].to(device)
                logits = mlp_model(sens)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                mlp_probs.append(p)
        mlp_probs = np.vstack(mlp_probs)
        mlp_preds = np.argmax(mlp_probs, axis=1)

        def mlp_pred_fn(x):
            with torch.no_grad():
                return mlp_model(x)
        lat = measure_inference_latency(mlp_pred_fn, (sample_sensor,))

        m = compute_classification_metrics(y_true, mlp_preds, mlp_probs, CLASS_NAMES)
        m["latency_ms_per_sample"] = lat
        results["Sensor-Only (MLP)"] = m
        probs_dict["Sensor-Only (MLP)"] = mlp_probs
    else:
        logger.warning("Sensor MLP checkpoint not found at %s. Skipping.", mlp_ckpt_path)

    # =========================================================================
    # 4. Multimodal Fusion (Feature-Level)
    # =========================================================================
    fusion_ckpt_path = os.path.join(checkpoints_dir, "multimodal_fusion_best.pt")
    if os.path.exists(fusion_ckpt_path):
        logger.info("Evaluating Multimodal Feature-Level Fusion...")
        fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device)
        f_cfg = fusion_ckpt.get("hyperparameters", {})
        fusion_model = MultimodalFusionNet(
            num_classes=len(CLASS_NAMES),
            num_sensor_features=len(feat_names),
            pretrained=False,
            image_embed_dim=f_cfg.get("image_embed_dim", 128),
            sensor_embed_dim=f_cfg.get("sensor_embed_dim", 32),
            fusion_hidden_dim=f_cfg.get("fusion_hidden_dim", 64),
        ).to(device)
        fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
        fusion_model.eval()

        fusion_probs = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)
                sens = batch["sensor"].to(device)
                logits = fusion_model(imgs, sens)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                fusion_probs.append(p)
        fusion_probs = np.vstack(fusion_probs)
        fusion_preds = np.argmax(fusion_probs, axis=1)

        def fusion_pred_fn(img_t, sens_t):
            with torch.no_grad():
                return fusion_model(img_t, sens_t)
        lat = measure_inference_latency(fusion_pred_fn, (sample_img, sample_sensor))

        m = compute_classification_metrics(y_true, fusion_preds, fusion_probs, CLASS_NAMES)
        m["latency_ms_per_sample"] = lat
        results["Multimodal Fusion (Feature-Level)"] = m
        probs_dict["Multimodal Fusion (Feature-Level)"] = fusion_probs
    else:
        logger.warning("Multimodal Fusion checkpoint not found at %s. Skipping.", fusion_ckpt_path)

    # =========================================================================
    # 5. Optional Late-Fusion
    # =========================================================================
    if "Image-Only (EfficientNet-B0)" in probs_dict and "Sensor-Only (MLP)" in probs_dict:
        logger.info("Evaluating Late Fusion (Probability Ensemble)...")
        late_fuser = LateFusionClassifier(image_weight=config.get("training", {}).get("late_fusion", {}).get("image_weight", 0.80))
        img_p_t = torch.tensor(probs_dict["Image-Only (EfficientNet-B0)"])
        sens_p_t = torch.tensor(probs_dict["Sensor-Only (MLP)"])
        late_p = late_fuser.combine_probabilities(img_p_t, sens_p_t).numpy()
        late_preds = np.argmax(late_p, axis=1)

        m = compute_classification_metrics(y_true, late_preds, late_p, CLASS_NAMES)
        m["latency_ms_per_sample"] = (
            results["Image-Only (EfficientNet-B0)"]["latency_ms_per_sample"] +
            results["Sensor-Only (MLP)"]["latency_ms_per_sample"]
        )
        results["Multimodal (Late Fusion)"] = m
        probs_dict["Multimodal (Late Fusion)"] = late_p

    # =========================================================================
    # Ablation Study: Multimodal vs Image-Only
    # =========================================================================
    ablation_report = {}
    if "Image-Only (EfficientNet-B0)" in results and "Multimodal Fusion (Feature-Level)" in results:
        img_res = results["Image-Only (EfficientNet-B0)"]
        fus_res = results["Multimodal Fusion (Feature-Level)"]

        delta_acc = round(fus_res["accuracy"] - img_res["accuracy"], 4)
        delta_macro_f1 = round(fus_res["macro_f1"] - img_res["macro_f1"], 4)
        delta_macro_recall = round(fus_res["macro_recall"] - img_res["macro_recall"], 4)

        ablation_report = {
            "image_only_macro_f1": img_res["macro_f1"],
            "multimodal_macro_f1": fus_res["macro_f1"],
            "delta_macro_f1": delta_macro_f1,
            "delta_accuracy": delta_acc,
            "delta_macro_recall": delta_macro_recall,
            "scientific_interpretation": (
                f"Feature-level fusion demonstrated a Macro F1 change of {delta_macro_f1:+.4f} "
                f"({delta_acc:+.4f} Accuracy) over the standalone image baseline. "
                "Because environmental sensor readings reflect canopy microclimate conditions "
                "rather than direct pathogen physical presence, environmental fusion provides "
                "contextual regularization, helping disambiguate borderline lesions under distinct humidity regimes."
            ),
        }

    # Save summary reports
    summary_path = os.path.join(metrics_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation summary to %s", summary_path)

    if ablation_report:
        ablation_path = os.path.join(metrics_dir, "ablation_study.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation_report, f, indent=2)
        logger.info("Saved ablation report to %s", ablation_path)

    # Generate Publication Plots
    plot_metric_comparisons(results, os.path.join(plots_dir, "metrics_comparison.png"))
    plot_confusion_matrices(results, os.path.join(plots_dir, "confusion_matrices.png"))
    plot_roc_curves(probs_dict, y_true, os.path.join(plots_dir, "roc_curves.png"))
    plot_loss_curves(metrics_dir, os.path.join(plots_dir, "loss_curves.png"))

    logger.info("================ FINAL TEST EVALUATION TABLE ================")
    for m_name, m_data in results.items():
        logger.info("%-35s | Acc: %.4f | Macro F1: %.4f | Latency: %.2f ms",
                    m_name, m_data["accuracy"], m_data["macro_f1"], m_data["latency_ms_per_sample"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    run_evaluation(args.config)
