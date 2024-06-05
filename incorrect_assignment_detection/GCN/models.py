import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import HGTConv

class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels) 

        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)

        return F.sigmoid(x)

# --------------------------------------------------------- #
# GCN MULTIPLE POST CONV LAYERS
# --------------------------------------------------------- #

# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import RGCNConv
# from torch_geometric.nn import HGTConv

# class GCNModel(nn.Module):
#     def __init__(self, num_features, hidden_channels):
#         super(GCNModel, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels) # 3
#         self.conv2 = GCNConv(hidden_channels, hidden_channels) # , 3
#         # self.conv3 = GCNConv(hidden_channels, hidden_channels-3) # , 3
#         self.fc1 = nn.Linear(hidden_channels, hidden_channels-3)
#         self.fc2 = nn.Linear(hidden_channels-3,hidden_channels-6)

#         self.fc = nn.Linear(hidden_channels-6, 1)
#         # Residual Connections
#     def forward(self, x, edge_index):

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         # x = self.conv3(x, edge_index)
#         # x = F.relu(x)
#         # x = F.dropout(x, training=self.training)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.fc2(x)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.fc(x)

#         return F.sigmoid(x)

