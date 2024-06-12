import glob
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    files = glob.glob(f'{data_dir}/*.pt')
    data_list = [torch.load(file) for file in files]
    return data_list

def split_data(data_list, train_size=0.8, valid_size=0.1, test_size=0.1):
    train_data, temp_data = train_test_split(data_list, train_size=train_size)
    valid_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + valid_size))
    return train_data, valid_data, test_data

def get_dataloaders(data_dir, batch_size=1, shuffle=True):
    data_list = load_data(data_dir)
    train_data, valid_data, test_data = split_data(data_list)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, valid_loader, test_loader
