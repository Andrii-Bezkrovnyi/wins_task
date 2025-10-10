from typing import Tuple
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train_torch_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epochs: int = 5
) -> None:
    """
    Train a PyTorch model for a given number of epochs.

    :param model: PyTorch model
    :param dataloader: Training DataLoader
    :param optimizer: Optimizer instance
    :param criterion: Loss function
    :param device: 'cpu' or 'cuda'
    :param epochs: Number of epochs
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}")


def evaluate_torch_model(
    model: nn.Module,
    X_test: Tensor,
    y_test: Tensor,
    device: str
) -> Tuple[float, Tensor]:
    """
    Evaluate a PyTorch model using accuracy.

    :param model: Trained PyTorch model
    :param X_test: Test features
    :param y_test: Test labels
    :param device: 'cpu' or 'cuda'
    :return: Tuple of (accuracy, predicted labels)
    """
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        predicts = outputs.argmax(dim=1).cpu()
    acc = accuracy_score(y_test.cpu(), predicts)
    return acc, predicts