from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .interface import MnistClassifierInterface


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    """

    def __init__(self) -> None:
        """
        Initialize the CNN architecture:
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 10).
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CNNClassifier(MnistClassifierInterface):
    """
    CNN classifier wrapper for MNIST dataset.
    """

    def __init__(
        self,
        lr: float = 0.001,
        epochs: int = 5,
        batch_size: int = 64,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the CNN classifier.

        Args:
            lr (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            device (Optional[str]): Device to run the model ('cuda' or 'cpu').
                                    If None, auto-selects 'cuda' if available.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = CNN().to(self.device)
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.optimizer: torch.optim.Optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion: nn.Module = nn.CrossEntropyLoss()

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Train the CNN model.

        Args:
            X_train (torch.Tensor): Training images tensor of shape (N, 1, 28, 28).
            y_train (torch.Tensor): Training labels tensor of shape (N,).
        """
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the input tensor.

        Args:
            X (torch.Tensor): Input images tensor of shape (N, 1, 28, 28).

        Returns:
            torch.Tensor: Predicted labels tensor of shape (N,).
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            return outputs.argmax(dim=1).cpu()
