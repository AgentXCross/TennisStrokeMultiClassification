from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

def get_dataloaders(batch_size = 32):
    """
    Gets dataloaders using `ImageFolder`. There should be one folder for all data.
    Within that folder should be the train and test sets. Within both the train and test sets,
    images should be in a corresponding class folder. Function applies transformations and
    creates DataLoaders where last is dropped.
    """
    torch.manual_seed(11)

    def center_crop_square(img: Image.Image) -> Image.Image:
        """Crops the center square from a PIL image."""
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        return img.crop((left, top, right, bottom))

    train_transform = transforms.Compose([
        transforms.Lambda(center_crop_square),
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Lambda(center_crop_square),     
        transforms.Resize((320, 320)),            
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root = "dataset/train_set", transform = train_transform)
    test_data = datasets.ImageFolder(root = "dataset/test_set", transform = test_transform)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 10)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True, num_workers = 10)

    return train_loader, test_loader