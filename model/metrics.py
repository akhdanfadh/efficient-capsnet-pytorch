import torch

def accuracy(y_true, y_pred):
    return torch.sum(y_true == y_pred).item() / len(y_true)