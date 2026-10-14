"""
Unit Tests for Machine Learning Architectures.
Tests forward passes, embedding projections, fusion, and agronomic risk estimation.
"""

import pytest
import numpy as np
import torch

from src.models.image_model import LeafDiseaseEfficientNet
from src.models.sensor_models import SensorMLP, SensorRandomForest
from src.models.fusion_model import MultimodalFusionNet, LateFusionClassifier
from src.models.risk_model import AgronomicRiskModel


def test_image_model_forward():
    model = LeafDiseaseEfficientNet(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, feats = model(x, return_features=True)
    assert logits.shape == (2, 3)
    assert feats.shape == (2, 1280)


def test_sensor_mlp_forward():
    model = SensorMLP(input_dim=5, hidden_dims=(64, 32), num_classes=3)
    x = torch.randn(4, 5)
    logits, embed = model(x, return_embedding=True)
    assert logits.shape == (4, 3)
    assert embed.shape == (4, 32)


def test_sensor_random_forest():
    rf = SensorRandomForest(n_estimators=10, max_depth=3, random_state=42)
    X = np.random.randn(20, 5)
    y = np.random.randint(0, 3, size=20)
    rf.fit(X, y)
    preds = rf.predict(X)
    probs = rf.predict_proba(X)
    assert len(preds) == 20
    assert probs.shape == (20, 3)


def test_multimodal_fusion_forward():
    model = MultimodalFusionNet(
        num_classes=3,
        num_sensor_features=5,
        pretrained=False,
        image_embed_dim=128,
        sensor_embed_dim=32,
        fusion_hidden_dim=64,
    )
    img = torch.randn(2, 3, 224, 224)
    sens = torch.randn(2, 5)
    logits, v_embed, s_embed = model(img, sens, return_embeddings=True)
    assert logits.shape == (2, 3)
    assert v_embed.shape == (2, 128)
    assert s_embed.shape == (2, 32)


def test_late_fusion_blender():
    late_fuser = LateFusionClassifier(image_weight=0.80)
    img_probs = torch.tensor([[0.7, 0.2, 0.1]])
    sens_probs = torch.tensor([[0.1, 0.8, 0.1]])
    combined = late_fuser.combine_probabilities(img_probs, sens_probs)
    assert combined.shape == (1, 3)
    expected = 0.80 * 0.7 + 0.20 * 0.1
    assert abs(combined[0, 0].item() - expected) < 1e-5


def test_agronomic_risk_model():
    risk_model = AgronomicRiskModel()
    
    # Severe conditions: high humidity (90%), prolonged leaf wetness (10h), warm temp (28°C)
    high_readings = {
        "ambient_temperature": 28.0,
        "soil_temperature": 26.0,
        "humidity": 90.0,
        "soil_moisture": 75.0,
        "leaf_wetness": 10.0,
    }
    high_res = risk_model.evaluate_risk(high_readings)
    assert high_res["risk_category"] == "HIGH"
    assert high_res["composite_risk_score"] > 65.0
    assert "disclaimer" in high_res

    # Low conditions: low humidity (40%), dry leaves (0h), temp 22°C
    low_readings = {
        "ambient_temperature": 22.0,
        "soil_temperature": 20.0,
        "humidity": 40.0,
        "soil_moisture": 35.0,
        "leaf_wetness": 0.0,
    }
    low_res = risk_model.evaluate_risk(low_readings)
    assert low_res["risk_category"] == "LOW"
    assert low_res["composite_risk_score"] < 35.0
