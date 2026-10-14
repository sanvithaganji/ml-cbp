"""
Visual Explainability via Grad-CAM (Gradient-weighted Class Activation Mapping).
Highlights regions in the leaf image that most influenced EfficientNet-B0 predictions.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM for EfficientNet-B0 visual architectures.
    Extracts gradients from the final convolutional block (features[-1]).
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        # If model is MultimodalFusionNet, access visual_backbone
        if hasattr(model, "visual_backbone"):
            self.backbone_features = model.visual_backbone
        elif hasattr(model, "features"):
            self.backbone_features = model.features
        else:
            raise AttributeError("Model must have 'features' or 'visual_backbone' attribute.")

        # Default to the final block of features
        self.target_layer = target_layer if target_layer is not None else self.backbone_features[-1]

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        sensor_tensor: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate 2D Grad-CAM heatmap array of shape (H, W) in range [0, 1].
        """
        self.model.zero_grad()

        # Forward pass
        if sensor_tensor is not None and hasattr(self.model, "sensor_branch"):
            logits = self.model(input_tensor, sensor_tensor)
        else:
            logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # Shape: (C, H, W)
        activations = self.activations[0]  # Shape: (C, H, W)

        # Global average pooling of gradients: alpha_k = (1 / Z) * sum(grad)
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)  # Shape: (H, W)
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions

        # Normalize between 0 and 1
        cam_np = cam.cpu().numpy()
        cam_max = np.max(cam_np)
        if cam_max > 0:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay_on_image(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Overlay the Grad-CAM heatmap onto the original PIL leaf image.
        
        Args:
            original_image: Original RGB PIL image
            heatmap: 2D numpy array [0, 1]
            alpha: Heatmap transparency factor
            colormap: Matplotlib colormap name
            
        Returns:
            Blended PIL Image
        """
        # Resize heatmap to match image dimensions
        orig_w, orig_h = original_image.size
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
        heatmap_arr = np.array(heatmap_resized) / 255.0

        # Apply colormap
        import matplotlib
        cmap = matplotlib.colormaps[colormap]
        colored_cam = cmap(heatmap_arr)[:, :, :3]  # Drop alpha channel
        colored_cam = np.uint8(255 * colored_cam)

        orig_arr = np.array(original_image.convert("RGB"))
        blended = np.uint8((1.0 - alpha) * orig_arr + alpha * colored_cam)

        return Image.fromarray(blended)
