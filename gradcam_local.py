from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np

def generate_all_gradcams(original_img: np.ndarray, image_tensor, model1, model2, model3):
    """Returns all 3 GradCAM Heatmaps."""
    cam_mb  = generate_gradcam(model1, "mobilenet", image_tensor, original_img)
    cam_res = generate_gradcam(model2, "resnet", image_tensor, original_img)
    cam_cx  = generate_gradcam(model3, "convnext", image_tensor, original_img)
    return cam_mb, cam_res, cam_cx

def generate_gradcam(model, model_name: str, image_tensor: torch.Tensor, original_img: np.ndarray):
    """Generate GradCAM Heatmap."""
    model.eval()
    target_layer = get_target_layer(model, model_name)
    
    cam = GradCAM(model = model, target_layers = [target_layer])
    grayscale_cam = cam(input_tensor = image_tensor, targets = None)[0] 
    visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb = True)

    return visualization  # returns a heatmap overlay (H, W, 3)

def get_target_layer(model, model_name: str):
    """Gets the target layer for applying GradCAM."""
    model_name = model_name.lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "convnext" in model_name:
        return model.features[-1]
    elif "mobilenet" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model type for Grad-CAM: {model_name}")