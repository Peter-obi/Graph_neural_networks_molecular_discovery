import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) model.
    """
    def __init__(self, input_dim, hidden_channels):
        """
        Initialize the GCN model.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_channels (int): Number of hidden channels.
        """
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Edge index tensor.
            batch (torch.Tensor): Batch tensor.
            edge_weight (torch.Tensor, optional): Edge weight tensor. Defaults to None.

        Returns:
            torch.Tensor: Output of the GCN model.
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)
