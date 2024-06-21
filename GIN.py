import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import MoleculeNet

class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) for predicting LogP.

    Attributes:
        conv1: First GIN convolution layer.
        conv2: Second GIN convolution layer.
        conv3: Third GIN convolution layer.
        conv4: Fourth GIN convolution layer.
        lin1: First linear layer.
        lin2: Second linear layer.
    """
    def __init__(self, num_node_features, dim):
        """
        Initialize the GIN model.

        Args:
            num_node_features (int): Number of node features.
            dim (int): Dimension of hidden layers.
        """
        super(GIN, self).__init__()
        self.conv1 = GINConv(Linear(num_node_features, dim))
        self.conv2 = GINConv(Linear(dim, dim))
        self.conv3 = GINConv(Linear(dim, dim))
        self.conv4 = GINConv(Linear(dim, dim))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, 1)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass for the GIN model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph edge indices.
            batch (torch.Tensor): Batch vector.

        Returns:
            torch.Tensor: Predicted LogP values.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# Load dataset
dataset = MoleculeNet(root='data', name='ESOL')

# Split dataset
torch.manual_seed(0)
dataset = dataset.shuffle()
train_dataset = dataset[:800]
val_dataset = dataset[800:900]
test_dataset = dataset[900:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(dataset.num_node_features, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

def train():
    """
    Train the GIN model.

    Returns:
        float: Average training loss.
    """
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def validate(loader):
    """
    Validate the GIN model.

    Args:
        loader (DataLoader): DataLoader for the validation/test set.

    Returns:
        float: Mean absolute error on the validation/test set.
    """
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            error += mean_absolute_error(output.cpu(), data.y.cpu()) * data.num_graphs
    return error / len(loader.dataset)

# Training loop
for epoch in range(1, 30):
    loss = train()
    val_error = validate(val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val MAE: {val_error:.4f}')

# Test the model
test_error = validate(test_loader)
print(f'Test MAE: {test_error:.4f}')
