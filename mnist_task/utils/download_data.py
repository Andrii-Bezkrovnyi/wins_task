from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

DATA_DIR = "./dataset"


def download_mnist() -> Tuple[MNIST, MNIST]:
    """
    Load the MNIST dataset with torchvision.
    
    Returns:
        trainset (Dataset): Training part of MNIST
        testset (Dataset): Test part of MNIST
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    return trainset, testset


def prepare_for_rf(
        trainset: MNIST,
        testset: MNIST
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for sklearn classifiers (Random Forest).
    Flatten images into 2D arrays.

    :trainset (Dataset): MNIST training dataset
    :testset (Dataset): MNIST test dataset
    
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    X_train = trainset.data.view(-1, 28 * 28).numpy()
    y_train = trainset.targets.numpy()
    X_test = testset.data.view(-1, 28 * 28).numpy()
    y_test = testset.targets.numpy()
    return X_train, y_train, X_test, y_test


def prepare_for_torch(
        trainset: MNIST,
        testset: MNIST
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare data as Tensors for PyTorch classifiers (NN, CNN).
    Normalize values to [0, 1].

    :trainset (Dataset): MNIST training dataset
    :testset (Dataset): MNIST test dataset
    
    Returns:
        X_train, y_train, X_test, y_test (torch tensors)
    """
    X_train = trainset.data.unsqueeze(1).float() / 255.0
    y_train = trainset.targets
    X_test = testset.data.unsqueeze(1).float() / 255.0
    y_test = testset.targets
    return X_train, y_train, X_test, y_test


def get_dataloaders(
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        batch_size: int = 64
) -> DataLoader:
    """
    Create DataLoader from tensors.

    Args:
        X_train (Tensor): Training features
        y_train (Tensor): Training labels
        batch_size (int): Batch size
    
    Returns:
        DataLoader
    """
    dataset = TensorDataset(X_train, y_train)
    downloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return downloader
