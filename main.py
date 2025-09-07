import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import TennisStrokeClassification
from dataset import get_dataloaders
from train_test_loop import train_test_loop
from extra_functions import accuracy_fn

#Device Agnostic Code
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

#Initialize Model, Optimizer, Scheduler, Loss Function
model = TennisStrokeClassification().to(device)
adam_optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001, weight_decay = 1e-4)
scheduler = StepLR(
    adam_optimizer,  
    step_size = 7,  
    gamma = 0.5    
)
loss_fn = nn.CrossEntropyLoss()

#Load the Datasets
train_loader, test_loader = get_dataloaders(batch_size = 32)

#Train and testing
train_test_loop(
    model = model,
    epochs = 51,
    device = device,
    optimizer = adam_optimizer,
    scheduling_function = scheduler,
    loss_function = loss_fn,
    train_dataloader = train_loader,
    test_dataloader = test_loader,
    accuracy_fn = accuracy_fn,
    seed = 72
)

# Save the trained model
model_path = "tennis_stroke_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at {model_path}.")