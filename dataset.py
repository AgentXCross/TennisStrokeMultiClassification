from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size = 32):
    """
    Gets dataloaders using `ImageFolder`. There should be one folder for all data.
    Within that folder should be the train and test sets. Within both the train and test sets,
    images should be in a corresponding class folder. Function applies transformations and
    creates DataLoaders where last is dropped.
    """
    train_transform = transforms.Compose([
        transforms.Resize(720),
        transforms.RandomCrop(720),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(720),
        transforms.CenterCrop(720),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root = "dataset/train_set", transform = train_transform)
    test_data = datasets.ImageFolder(root = "data/test_set", transform = test_transform)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, test_loader