import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def accuracy_fn(pred, true):
    correct = torch.eq(pred, true).sum().item()
    return correct / len(pred) * 100

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