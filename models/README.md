# Models Directory

This directory stores trained model checkpoints, architecture configurations, and fitted feature preprocessors.

## Subdirectories
- `checkpoints/`: PyTorch model weights (`.pt` / `.pth`) and scikit-learn models (`.joblib`):
  - `image_efficientnet_best.pt`: Image-only baseline checkpoint
  - `sensor_rf_best.joblib`: Random Forest baseline model
  - `sensor_mlp_best.pt`: Sensor MLP baseline checkpoint
  - `multimodal_fusion_best.pt`: End-to-end multimodal fusion checkpoint
- `preprocessors/`:
  - `sensor_scaler.joblib`: StandardScaler and SimpleImputer fitted strictly on training split data

## Checkpoint Format
PyTorch checkpoints contain a dictionary with:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Best validation epoch
- `metrics`: Dictionary of evaluation metrics on validation set
- `classes`: Ordered class list `["Healthy", "Anthracnose", "Leaf Spot"]`
- `class_to_idx`: `{"Healthy": 0, "Anthracnose": 1, "Leaf Spot": 2}`
- `feature_names`: Sensor feature names in exact column order
- `architecture`: Model hyperparameter configuration
