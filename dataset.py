import torch
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from typing import Callable, List
from torch.utils.data import ConcatDataset, DataLoader
from PIL import Image
import numpy as np

label_map = {
    'backhand': 0,
    'forehand': 1,
    'ready_position': 2,
    'serve': 3
}

class TennisStrokeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.transforms, label_map: dict):
        """
        df: Pandas DataFrame containing 'filepath' and 'label'
        transform: Albumentations transform
        label_map: dict mapping label string -> int
        """
        self.df = df.reset_index(drop = True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image (H, W, C) in BGR format
        img = cv2.imread(row['filepath'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Apply transformations
        img = self.transform(image = img)['image']

        # Convert label from string to int
        label = self.label_map[row['label']]
        label = torch.tensor(label, dtype = torch.long)

        return img, label
    
def create_image_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transforms: List[Callable[[Image.Image], torch.Tensor]],
    test_transform: Callable[[Image.Image], torch.Tensor],
    label_map: dict,
    batch_size: int = 32,
    seed: int = 73,
):
    """
        Creates a training and testing dataset from 1 Pandas Dataframe

        Params:
            train_df: Training Pandas DataFrame containing ['filepath', 'filename', 'label'] columns
            test_df: Testing Pandas DataFrame containing ['filepath', 'filename', 'label'] columns
            train_transforms: List of training set transformations
            test_transform: Testing set transformations
            batch_size: DataLoader batch size, default = 32
            seed: Manual Seed value, default = 73

        Returns 2 DataLoaders
            train_dataloader, test_dataloader
    """

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create training datasets for each tranform and concatenate the sets
    train_datasets = [
        TennisStrokeDataset(
            df = train_df,
            transform = tfm,
            label_map = label_map
        ) for tfm in train_transforms
    ]

    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = train_datasets[0]

    # Testing Set
    test_dataset = TennisStrokeDataset(
            df = test_df,
            transform = test_transform,
            label_map = label_map
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = False)

    return train_loader, test_loader