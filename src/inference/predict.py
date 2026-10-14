"""
Unified Inference Engine for Leaf Disease Prediction.
Supports Image-Only, Sensor-Only, Multimodal Feature-Level Fusion, and Late Fusion.
Transparently handles missing sensors and provides risk assessment without fabricated data.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.reproducibility import get_device
from src.data.preprocessing import get_image_transforms, EnvironmentalPreprocessor
from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX, FEATURE_NAMES
from src.models.image_model import LeafDiseaseEfficientNet
from src.models.sensor_models import SensorMLP, SensorRandomForest
from src.models.fusion_model import MultimodalFusionNet, LateFusionClassifier
from src.models.risk_model import AgronomicRiskModel

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """
    Unified predictor supporting multimodal, image-only, and sensor-only inference.
    """

    def __init__(
        self,
        checkpoints_dir: str = "models/checkpoints",
        preprocessor_dir: str = "models/preprocessors",
        device_preference: str = "auto",
    ):
        self.checkpoints_dir = checkpoints_dir
        self.preprocessor_dir = preprocessor_dir
        self.device = get_device(device_preference)
        self.image_transforms = get_image_transforms(input_size=224, is_training=False)
        self.risk_model = AgronomicRiskModel()

        # Lazy loaded models
        self.image_model: Optional[LeafDiseaseEfficientNet] = None
        self.sensor_rf: Optional[SensorRandomForest] = None
        self.sensor_mlp: Optional[SensorMLP] = None
        self.fusion_model: Optional[MultimodalFusionNet] = None
        self.late_fuser = LateFusionClassifier(image_weight=0.80)
        self.preprocessor: Optional[EnvironmentalPreprocessor] = None

        self._load_preprocessor()

    def _load_preprocessor(self):
        preproc_path = os.path.join(self.preprocessor_dir, "sensor_scaler.joblib")
        if os.path.exists(preproc_path):
            try:
                self.preprocessor = EnvironmentalPreprocessor.load(preproc_path)
            except Exception as e:
                logger.warning("Could not load preprocessor: %s", str(e))

    def load_image_model(self) -> LeafDiseaseEfficientNet:
        if self.image_model is None:
            ckpt_path = os.path.join(self.checkpoints_dir, "image_efficientnet_best.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Image model checkpoint not found at {ckpt_path}. Train the model first.")
            model = LeafDiseaseEfficientNet(num_classes=len(CLASS_NAMES), pretrained=False).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.image_model = model
        return self.image_model

    def load_sensor_rf(self) -> SensorRandomForest:
        if self.sensor_rf is None:
            rf_path = os.path.join(self.checkpoints_dir, "sensor_rf_best.joblib")
            if not os.path.exists(rf_path):
                raise FileNotFoundError(f"Sensor RF model not found at {rf_path}. Train the model first.")
            self.sensor_rf = SensorRandomForest.load(rf_path)
        return self.sensor_rf

    def load_sensor_mlp(self) -> SensorMLP:
        if self.sensor_mlp is None:
            mlp_path = os.path.join(self.checkpoints_dir, "sensor_mlp_best.pt")
            if not os.path.exists(mlp_path):
                raise FileNotFoundError(f"Sensor MLP checkpoint not found at {mlp_path}. Train the model first.")
            ckpt = torch.load(mlp_path, map_location=self.device)
            model = SensorMLP(
                input_dim=len(FEATURE_NAMES),
                hidden_dims=tuple(ckpt.get("hidden_dims", (64, 32))),
                num_classes=len(CLASS_NAMES),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.sensor_mlp = model
        return self.sensor_mlp

    def load_fusion_model(self) -> MultimodalFusionNet:
        if self.fusion_model is None:
            ckpt_path = os.path.join(self.checkpoints_dir, "multimodal_fusion_best.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Fusion model checkpoint not found at {ckpt_path}. Train the model first.")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            f_cfg = ckpt.get("hyperparameters", {})
            model = MultimodalFusionNet(
                num_classes=len(CLASS_NAMES),
                num_sensor_features=len(FEATURE_NAMES),
                pretrained=False,
                image_embed_dim=f_cfg.get("image_embed_dim", 128),
                sensor_embed_dim=f_cfg.get("sensor_embed_dim", 32),
                fusion_hidden_dim=f_cfg.get("fusion_hidden_dim", 64),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.fusion_model = model
        return self.fusion_model

    def predict(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        sensor_dict: Optional[Dict[str, float]] = None,
        mode: str = "multimodal",
    ) -> Dict[str, Any]:
        """
        Execute prediction pipeline.
        
        Args:
            image: PIL Image instance or file path string.
            sensor_dict: Dict with keys ['ambient_temperature', 'soil_temperature', 'humidity', 'soil_moisture', 'leaf_wetness']
            mode: 'multimodal', 'image_only', 'sensor_only_rf', 'sensor_only_mlp', 'late_fusion'
            
        Returns:
            Dictionary containing prediction results, probabilities, confidence, and risk score.
        """
        start_time = time.perf_counter()
        warning_msg = None
        fallback_occurred = False

        # Load image if string path provided
        pil_img = None
        if isinstance(image, str):
            if os.path.exists(image):
                pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")

        # Validate sensor availability
        has_sensors = sensor_dict is not None and all(
            feat in sensor_dict and sensor_dict[feat] is not None for feat in FEATURE_NAMES
        )

        # Handle missing modalities and mode adjustments
        if mode in ("multimodal", "late_fusion") and not has_sensors:
            if pil_img is not None:
                mode = "image_only"
                fallback_occurred = True
                warning_msg = "Environmental sensor readings not provided. Safely fell back to Image-Only baseline without fabricating sensor inputs."
            else:
                raise ValueError("Neither valid image nor complete sensor readings were supplied.")

        if mode == "image_only" and pil_img is None:
            raise ValueError("Image-only mode requested but no image was provided.")

        if mode in ("sensor_only_rf", "sensor_only_mlp") and not has_sensors:
            raise ValueError(f"Sensor-only mode ({mode}) requested but sensor readings are incomplete or missing.")

        # Transform inputs
        img_tensor = None
        if pil_img is not None:
            img_tensor = self.image_transforms(pil_img).unsqueeze(0).to(self.device)

        sensor_tensor = None
        if has_sensors and self.preprocessor is not None:
            sensor_vec = self.preprocessor.transform_single(sensor_dict)
            sensor_tensor = torch.from_numpy(sensor_vec).float().unsqueeze(0).to(self.device)

        # Execute selected model
        probs = None
        actual_model_used = mode

        with torch.no_grad():
            if mode == "multimodal":
                model = self.load_fusion_model()
                logits = model(img_tensor, sensor_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                actual_model_used = "Multimodal Fusion (Feature-Level)"

            elif mode == "image_only":
                model = self.load_image_model()
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                actual_model_used = "Image-Only (EfficientNet-B0)"

            elif mode == "sensor_only_rf":
                model = self.load_sensor_rf()
                sensor_vec = self.preprocessor.transform_single(sensor_dict)
                probs = model.predict_proba(sensor_vec.reshape(1, -1))[0]
                actual_model_used = "Sensor-Only (Random Forest)"

            elif mode == "sensor_only_mlp":
                model = self.load_sensor_mlp()
                logits = model(sensor_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                actual_model_used = "Sensor-Only (PyTorch MLP)"

            elif mode == "late_fusion":
                img_model = self.load_image_model()
                mlp_model = self.load_sensor_mlp()
                img_logits = img_model(img_tensor)
                sens_logits = mlp_model(sensor_tensor)
                p_img = torch.softmax(img_logits, dim=1)
                p_sens = torch.softmax(sens_logits, dim=1)
                p_combined = self.late_fuser.combine_probabilities(p_img, p_sens)
                probs = p_combined.cpu().numpy()[0]
                actual_model_used = "Multimodal (Late Fusion Ensemble)"

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Environmental risk assessment
        risk_assessment = None
        if has_sensors:
            risk_assessment = self.risk_model.evaluate_risk(sensor_dict)

        prob_breakdown = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_breakdown,
            "model_used": actual_model_used,
            "fallback_occurred": fallback_occurred,
            "warning": warning_msg,
            "has_sensor_data": has_sensors,
            "risk_assessment": risk_assessment,
            "latency_ms": round(latency_ms, 2),
        }
