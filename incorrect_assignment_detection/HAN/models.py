import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv

class HANModel(nn.Module):
    def __init__(self, num_features, hidden_channels, metadata):
        super(HANModel, self).__init__()

        self.conv1 = HANConv(num_features, hidden_channels, metadata, 3) 
        self.conv2 = HANConv(hidden_channels, hidden_channels, metadata, 3) 

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index):

        x = self.conv1(x_dict, edge_index)
        x = x['papers']
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = {
            "papers": x
        }
        x = self.conv2(x, edge_index)
        x = x['papers']
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)

        return F.sigmoid(x)
    
    