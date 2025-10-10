from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .interface import MnistClassifierInterface


class SimpleNN(nn.Module):
    """
    A simple feed-forward neural network for MNIST classification.
    """
    def __init__(
            self,
            input_dim: int = 28 * 28,
            hidden1: int = 128,
            hidden2: int = 64,
            num_classes: int = 10
    ) -> None:
        """
        Initialize the simple neural network architecture.

        Args:
            input_dim (int): Number of input features (default is 28*28 for MNIST).
            hidden1 (int): Number of units in the first hidden layer.
            hidden2 (int): Number of units in the second hidden layer.
            num_classes (int): Number of output classes (default: 10).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        return self.net(x)


class FeedForwardNN(MnistClassifierInterface):
    """
    Feed-forward neural network classifier for MNIST with early stopping.
    """
    def __init__(
        self,
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64,
        patience: int = 5,
        min_delta: float = 1e-4,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the FeedForwardNN classifier.

        Args:
            lr (float): Learning rate for optimizer
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Number of epochs to wait for improvement before early stopping
            min_delta (float): Minimum improvement in validation loss to reset patience
            device (Optional[str]): Device to run the model on ('cuda' or 'cpu').
                                    Auto-selects CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN().to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Train the model with early stopping using a validation split.

        Args:
            X_train (torch.Tensor): Training features, shape (N, 1, 28, 28)
            y_train (torch.Tensor): Training labels, shape (N,)
        """
        # Split into train and validation sets
        val_size = int(0.1 * len(X_train))
        train_size = len(X_train) - val_size
        train_ds, val_ds = random_split(
            TensorDataset(X_train, y_train),
            [train_size, val_size]
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Validation step
            val_loss = self._validate(val_loader)

            # Check for improvement
            if val_loss + self.min_delta < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Early stopping
            if no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

    def _validate(self, loader: DataLoader) -> float:
        """
        Compute the average validation loss.

        Args:
            loader (DataLoader): Validation data loader

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for the input tensor.

        Args:
            X (torch.Tensor): Input tensor, shape (N, 1, 28, 28)

        Returns:
            torch.Tensor: Predicted labels, shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            return outputs.argmax(dim=1).cpu()