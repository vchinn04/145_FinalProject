import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCNModel, self).__init__()

        self.conv1 = RGCNConv(num_features, hidden_channels, 3) 
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, 3) 

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_type):

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)

        return F.sigmoid(x)