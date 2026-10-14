"""
Reproducibility and Device Selection Utilities.
Ensures deterministic random states across Python, NumPy, and PyTorch.
Provides automated hardware device detection (Apple Silicon MPS, NVIDIA CUDA, CPU).
"""

import os
import random
import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set seed across all pseudo-random number generators for strict reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS uses torch.manual_seed
        pass

    logger.info("Deterministic seed set to %d", seed)


def get_device(preference: str = "auto") -> torch.device:
    """
    Select computing device based on preference and available hardware.
    
    Order of auto-resolution:
    1. Apple Silicon MPS (Metal Performance Shaders on macOS)
    2. NVIDIA CUDA
    3. CPU fallback
    
    Args:
        preference: "auto", "mps", "cuda", or "cpu"
        
    Returns:
        torch.device
    """
    preference = (preference or "auto").lower()

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    # Auto resolution
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
