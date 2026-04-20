"""
Data loading utilities for CIFAR-100 dataset
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from config import (
    DATASET_PATH, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, 
    CORESET_FRACTION
)


def get_cifar100_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                         pin_memory=PIN_MEMORY, train_fraction=1.0, 
                         coreset_indices=None):
    """
    Load CIFAR-100 dataset with train/test splits
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster transfer to GPU
        train_fraction: Fraction of training data to use
        coreset_indices: Optional indices for coreset training
    
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        train_dataset: Full training dataset
        test_dataset: Test dataset
    """
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    # Download and load CIFAR-100
    train_dataset = datasets.CIFAR100(
        root=DATASET_PATH,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = datasets.CIFAR100(
        root=DATASET_PATH,
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Apply coreset or fraction sampling
    if coreset_indices is not None:
        # Use specific coreset indices
        train_dataset = Subset(train_dataset, coreset_indices)
    elif train_fraction < 1.0:
        # Random sampling
        n_samples = int(len(train_dataset) * train_fraction)
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_full_dataset(split='train'):
    """Get full CIFAR-100 dataset without transforms for RES calculation"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    if split == 'train':
        dataset = datasets.CIFAR100(
            root=DATASET_PATH,
            train=True,
            transform=transform,
            download=True
        )
    else:
        dataset = datasets.CIFAR100(
            root=DATASET_PATH,
            train=False,
            transform=transform,
            download=True
        )
    
    return dataset
