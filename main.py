from model_definition import FocalLoss, create_convnext_tiny, create_mobilenet_v3, create_resnet18
from helper_functions import accuracy_function
from dataset import create_image_dataloaders
from train_test_loop import training_testing_loop
from transforms import transforms
import torch
import pandas as pd

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Import dataframes
train_df = pd.read_csv('dataframe/train_df.csv')
test_df = pd.read_csv('dataframe/test.csv')

# Label mapping
label_map = {
    'backhand': 0,
    'forehand': 1,
    'ready_position': 2,
    'serve': 3
}

# Get transformations
train_transforms, test_transform = transforms()

# Get image dataloaders
train_dataloader, test_dataloader = create_image_dataloaders(
    train_df = train_df,
    test_df = test_df,
    train_transforms = train_transforms,
    test_transform = test_transform,
    label_map = label_map
)

# Initialize Model, Loss Function, and Optimizer
mobilenet_v3_model = create_mobilenet_v3(num_classes = 4)
resnet18_model = create_resnet18(num_classes = 4)
convnext_model = create_convnext_tiny(num_classes = 4)

mobilenet_v3_model.to(device)
resnet18_model.to(device)
convnext_model.to(device)

adam_optimizer_mobilenetv3 = torch.optim.Adam(params = mobilenet_v3_model.parameters(), lr = 1e-4)
adam_optimizer_resnet18 = torch.optim.Adam(params = resnet18_model.parameters(), lr = 1e-4)
adam_optimizer_convnext = torch.optim.Adam(params = convnext_model.parameters(), lr = 1e-4)

loss_fn = FocalLoss(alpha = [1.0, 1.0, 1.0, 1.0], gamma = 2.0)

# Training
if __name__ == '__main__':
    print('MobileNetV3-Small Model:')
    training_testing_loop(
        mobilenet_v3_model, 
        'mobilenet_v3',
        train_dataloader, 
        test_dataloader, 
        device, 
        max_epochs = 6, 
        optimizer = adam_optimizer_mobilenetv3,
        loss_function = loss_fn,
        accuracy_function = accuracy_function
    )
    print('\n')
    print('ResNet18 Model')
    training_testing_loop(
        resnet18_model, 
        'resnet18',
        train_dataloader, 
        test_dataloader, 
        device, 
        max_epochs = 6, 
        optimizer = adam_optimizer_resnet18,
        loss_function = loss_fn,
        accuracy_function = accuracy_function
    )
    print('\n')
    print('ConvNeXt-Tiny Model')
    convnext_results = training_testing_loop(
        convnext_model, 
        'convnext',
        train_dataloader, 
        test_dataloader, 
        device, 
        max_epochs = 6, 
        optimizer = adam_optimizer_convnext,
        loss_function = loss_fn,
        accuracy_function = accuracy_function
    )

