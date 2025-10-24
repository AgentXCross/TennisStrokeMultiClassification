import torch
from typing import Callable
from helper_functions import accuracy_function, save_model_weights

def training_testing_loop(
        model: torch.nn.Module,
        model_name: str,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: str,
        max_epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.Module,
        accuracy_function: Callable[[torch.Tensor, torch.Tensor], float] = accuracy_function,
        seed: int = 73,
) -> dict:
    """
    Training and Testing Loop

    Args:
        model: PyTorch model
        model_name: Name of the model (for saving weights)
        train_dataloader: Dataloader containing training set
        test_dataloader: Dataloader containing testation set
        device: cuda or cpu or mps
        max_epochs: number of epochs
        optimizer: Optimization Functions
        accuracy_function: Calculates accuracy
        optimzer: Optimization function
        seed: For reproducibility
    """
    # Set seed
    torch.manual_seed(seed)

    # 4 Lists to keep track of train/test accuracy/loss.
    train_loss, test_loss, train_accuracy, test_accuracy = [], [], [], []

    # Move model to proper device
    model.to(device)

    # Keep track of the best test loss
    best_test_loss = float("inf")

    for epoch in range(max_epochs):
        # ------------------------ Training ------------------------------
        model.train()

        # Keep track of total loss and accuracy and divide by batch size at the end of every epoch
        total_train_loss, total_train_accuracy = 0, 0

        for X_batch_images, y_batch_labels in train_dataloader:
            X_batch_images = X_batch_images.to(device)
            
            y_batch_labels = y_batch_labels.long().to(device)

            optimizer.zero_grad()
            y_train_pred_logits = model(X_batch_images)
            loss = loss_function(y_train_pred_logits, y_batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_accuracy += accuracy_function(y_batch_labels.squeeze(), y_train_pred_logits)

        # Calculate train loss and accuracy over the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_accuracy / len(train_dataloader)

        # ----------------------- Testing ---------------------------
        # Set model to evaluation mode for testing
        model.eval()
        total_test_loss, total_test_accuracy = 0, 0
        test_preds, test_targets = [], []

        with torch.inference_mode():
            for X_batch_images, y_batch_labels in test_dataloader:
                X_batch_images = X_batch_images.to(device)

                y_batch_labels = y_batch_labels.long().to(device)

                y_test_pred_logits = model(X_batch_images)
                loss = loss_function(y_test_pred_logits, y_batch_labels)
                total_test_loss += loss.item()
                total_test_accuracy += accuracy_function(y_batch_labels.squeeze(), y_test_pred_logits)

                preds = torch.argmax(y_test_pred_logits, dim = 1)
                test_preds.extend(preds.cpu().tolist())
                test_targets.extend(y_batch_labels.cpu().tolist())

            avg_test_loss = total_test_loss / len(test_dataloader)
            avg_test_acc = total_test_accuracy / len(test_dataloader)

        print(f"========== Epoch [{epoch + 1}/{max_epochs}] ============")
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print(f"Train Acc:  {avg_train_acc:.4f}% | Test Acc:  {avg_test_acc:.4f}%\n")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_model_weights(model, filename = f"{model_name}.pth")
            print(f"✅ Best model saved at epoch {epoch+1} with test loss {avg_test_loss:.4f}")

        # Append losses and accuracies
        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)
        train_accuracy.append(avg_train_acc)
        test_accuracy.append(avg_test_acc)

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_preds": test_preds,
        "test_targets": test_targets
    }