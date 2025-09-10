import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import TennisStrokeClassifier
from dataset import get_dataloaders
from train_test_loop import train_test_loop
from extra_functions import accuracy_fn

#Device Agnostic Code
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#Initialize Model, Optimizer, Scheduler, Loss Function
model = TennisStrokeClassifier().to(device)
#Adjust the sgd_optimizer tomorrow
sgd_optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 1e-4)

sgd_scheduler = StepLR(
    sgd_optimizer,  
    step_size = 7,  
    gamma = 0.9    
)

loss_fn = nn.CrossEntropyLoss()

#Load the Datasets
train_loader, test_loader = get_dataloaders(batch_size = 32)

#Train and testing
train_test_loop(
    model = model,
    epochs = 51,
    device = device,
    optimizer = sgd_optimizer,
    scheduling_function = sgd_scheduler,
    loss_function = loss_fn,
    train_dataloader = train_loader,
    test_dataloader = test_loader,
    accuracy_fn = accuracy_fn,
    seed = 11
)

# Save the trained model
model_path = "tennis_stroke_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at {model_path}.")