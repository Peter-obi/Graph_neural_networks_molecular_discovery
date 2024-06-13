import glob
import os
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_dirs):
    """
    Load data from multiple directories containing .pt files.

    Args:
        data_dirs (list): List of directory paths containing .pt files.

    Returns:
        list: List of loaded data objects.
    """
    data_list = []
    for data_dir in data_dirs:
        files = glob.glob(f'{data_dir}/*.pt')
        data_list.extend([torch.load(file) for file in files])
    return data_list

def split_data(data_list, train_size=0.8, valid_size=0.1, test_size=0.1):
    """
    Split data into train, validation, and test sets.

    Args:
        data_list (list): List of data objects.
        train_size (float): Proportion of data to use for training.
        valid_size (float): Proportion of data to use for validation.
        test_size (float): Proportion of data to use for testing.

    Returns:
        tuple: Train, validation, and test data sets.
    """
    train_data, temp_data = train_test_split(data_list, train_size=train_size)
    valid_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + valid_size))
    return train_data, valid_data, test_data

def get_dataloaders(data_dirs, batch_size=1, shuffle=True):
    """
    Create data loaders for train, validation, and test sets.

    Args:
        data_dirs (list): List of directory paths containing .pt files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        tuple: Train, validation, and test data loaders.
    """
    data_list = load_data(data_dirs)
    train_data, valid_data, test_data = split_data(data_list)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, valid_loader, test_loader
