import torch

def accuracy(y_pred, y_true):
    return torch.sum(y_true == y_pred).item() / len(y_true)