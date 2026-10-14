"""
Evaluation Metrics Utility.
Calculates Accuracy, Macro P/R/F1, Per-class metrics, Confusion Matrix,
One-vs-Rest ROC-AUC, and Inference Latency per sample.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth class indices of shape (N,)
        y_pred: Predicted class indices of shape (N,)
        y_prob: Predicted probability distributions of shape (N, C), optional
        class_names: Names corresponding to class indices 0..C-1
        
    Returns:
        Dictionary containing overall and per-class metrics.
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class_metrics = {}
    for idx, cname in enumerate(class_names):
        p_val = float(per_class_p[idx]) if idx < len(per_class_p) else 0.0
        r_val = float(per_class_r[idx]) if idx < len(per_class_r) else 0.0
        f_val = float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        support = int(np.sum(y_true == idx))
        per_class_metrics[cname] = {
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1_score": round(f_val, 4),
            "support": support,
        }
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # ROC-AUC (One-vs-Rest) if probabilities are provided
    roc_auc = None
    if y_prob is not None and y_prob.shape[1] == len(class_names):
        try:
            # Check if all classes are present in y_true
            if len(np.unique(y_true)) == len(class_names):
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None
            
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc_ovr": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }


def measure_inference_latency(
    predict_fn,
    sample_inputs: Tuple,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> float:
    """
    Measure average inference latency per sample in milliseconds.
    
    Args:
        predict_fn: Callable that accepts *sample_inputs
        sample_inputs: Tuple of inputs to pass to predict_fn
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark runs
        
    Returns:
        Latency in milliseconds per sample.
    """
    for _ in range(num_warmup):
        _ = predict_fn(*sample_inputs)
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = predict_fn(*sample_inputs)
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    return round(latency_ms, 2)
