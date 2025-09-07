import torch

# Wrapper Function
def train_test_loop(
        model: torch.nn.Module,
        epochs: int,
        device: str,
        optimizer: torch.optim.Optimizer,
        scheduling_function: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        accuracy_fn,
        seed: int = 73

):
    """
    Runs `train_step()` and `test_step()`. Together, they run one epoch (forward pass, backpropagation).
    """
    model = model.to(device)
    torch.manual_seed(seed)
    for epoch in range(epochs):
        print(f"Epoch: {epoch} ==============================")
        model.train()
        train_step(
            model = model,
            dataloader = train_dataloader,
            loss_function = loss_function,
            optimization_function = optimizer,
            accuracy_function = accuracy_fn,
            device = device
        )
        model.eval()
        with torch.inference_mode():
            test_step(
                model = model,
                loss_function = loss_function,
                accuracy_function = accuracy_fn,
                dataloader = test_dataloader,
                device = device
            )
            scheduling_function.step()

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_function: torch.nn.Module,
        optimization_function: torch.optim.Optimizer,
        accuracy_function,
        device: str
):
    """
    One train step. Takes a dataloader. Performs a forward pass, calculates loss/accuracy, and backpropagates.
    Dataloader must drop last.
    """
    train_loss_total, train_acc_total = 0, 0
    for X_batch, y_batch in dataloader:
        # Move to best device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Forward pass
        y_train_preds_logits = model(X_batch)
        # Loss
        loss = loss_function(y_train_preds_logits, y_batch)
        train_loss_total += loss.item()
        # Backpropagation
        optimization_function.zero_grad()
        loss.backward()
        # Gradient Descent
        optimization_function.step()
        # Accuracy
        accuracy = accuracy_function(y_train_preds_logits.argmax(dim = 1), y_batch)
        train_acc_total += accuracy
    train_acc = train_acc_total / len(dataloader)
    train_loss = train_loss_total / len(dataloader)
    print(f"Train Loss: {train_loss} | Train Accuracy: {train_acc}")

def test_step(
        model: torch.nn.Module,
        loss_function: torch.nn.Module,
        accuracy_function,
        dataloader: torch.utils.data.DataLoader,
        device: str
):
    """
    One test step. Takes a dataloader. Performs a forward pass, calculates loss and accuracy.
    Dataloader must drop last.
    """
    test_loss_total, test_accuracy_total = 0, 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # Forward pass
        y_test_preds_logits = model(X_batch)
        # Loss
        loss = loss_function(y_test_preds_logits, y_batch)
        test_loss_total += loss.item()
        # Accuracy
        accuracy = accuracy_function(y_test_preds_logits.argmax(dim = 1), y_batch)
        test_accuracy_total += accuracy
    test_acc = test_accuracy_total / len(dataloader)
    test_loss = test_loss_total / len(dataloader)
    print(f"Test Loss: {test_loss} | Test Accuracy: {test_acc}")
