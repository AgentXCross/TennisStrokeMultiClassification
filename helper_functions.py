import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

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

def save_model_weights(model: nn.Module, filename: str, save_dir: str = 'results-models') -> None:
    """
    Saves a PyTorch model's weights (state_dict) to a specified directory.

    Args:
        model (nn.Module): Trained PyTorch model
        save_dir (str): Directory to save model weights into (default: results-models)
        filename (str): Filename for the weights
    """
    os.makedirs(save_dir, exist_ok = True)

    filepath = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), filepath)

def accuracy_function(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    """
    Computes accuracy for multiclass tasks.
    Args:
        y_true: Ground truth labels
        y_pred_logits: Model logit outputs
    """
    preds = torch.argmax(y_pred_logits, dim = 1)  # shape (N,)
    return (preds == y_true).float().mean().item() * 100

label_map = {
    'backhand': 0,
    'forehand': 1,
    'ready_position': 2,
    'serve': 3
}
idx_to_class = {v: k for k, v in label_map.items()}
def unnormalize(img_tensor: torch.Tensor) -> Image.Image:
    """Unnormalize Tensor Values."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  
    img = (img * std) + mean           
    img = np.clip(img, 0, 1)      
    return img

def visualize_batch(data_loader: torch.utils.data.DataLoader, num_images = 4):
    """Visualize 4 Images from a Dataloader"""
    images, labels = next(iter(data_loader))

    plt.figure(figsize = (14, 4))
    for i in range(num_images):
        img = unnormalize(images[i])
        label = idx_to_class[labels[i].item()] 

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

    plt.show()

def show_random_images(folder_path: str, num_images: int = 4) -> None:
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    sample_files = random.sample(files, num_images)

    plt.figure(figsize = (12, 4))
    for i, filename in enumerate(sample_files):
        img = Image.open(os.path.join(folder_path, filename))
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(filename[:15])
        plt.axis("off")
    plt.show()