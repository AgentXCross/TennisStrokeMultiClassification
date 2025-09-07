import torch
def accuracy_fn(pred, true):
    correct = torch.eq(pred, true).sum().item()
    return correct / len(pred) * 100