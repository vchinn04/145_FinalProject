import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(RGCNModel, self).__init__()

        self.conv1 = RGCNConv(num_features, hidden_channels, 3) 
        self.conv2 = RGCNConv(hidden_channels, hidden_channels//2, 3) 
        self.conv3 = RGCNConv(hidden_channels//2, hidden_channels//4, 3) 
        self.conv4 = RGCNConv(hidden_channels//4, hidden_channels//4, 3) 

        self.fc = nn.Linear(hidden_channels//4, 1)

    def forward(self, x, edge_index, edge_type):

        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)


        x = self.fc(x)

        return F.sigmoid(x)
    
# --------------------------------------------------------- #
# RGCN 5 CONVOLUTION LAYERS
# --------------------------------------------------------- #
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import RGCNConv

# class RGCNModel(nn.Module):
#     def __init__(self, num_features, hidden_channels):
#         super(RGCNModel, self).__init__()

#         self.conv1 = RGCNConv(num_features, hidden_channels, 3) 
#         self.conv2 = RGCNConv(hidden_channels, hidden_channels//2, 3) 
#         self.conv3 = RGCNConv(hidden_channels//2, hidden_channels//4, 3) 
#         self.conv4 = RGCNConv(hidden_channels//4, hidden_channels//6, 3) 
#         self.conv5 = RGCNConv(hidden_channels//6, hidden_channels//6, 3) 

#         self.fc = nn.Linear(hidden_channels//6, 1)

#     def forward(self, x, edge_index, edge_type):

#         x = self.conv1(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv2(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv3(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv4(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv5(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)


#         x = self.fc(x)

#         return F.sigmoid(x)

# --------------------------------------------------------- #
# RGCN 2 CONVOLUTION LAYERS
# --------------------------------------------------------- #
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import RGCNConv

# class RGCNModel(nn.Module):
#     def __init__(self, num_features, hidden_channels):
#         super(RGCNModel, self).__init__()

#         self.conv1 = RGCNConv(num_features, hidden_channels, 3) 
#         self.conv2 = RGCNConv(hidden_channels, hidden_channels, 3) 

#         self.fc = nn.Linear(hidden_channels, 1)

#     def forward(self, x, edge_index, edge_type):

#         x = self.conv1(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.conv2(x, edge_index, edge_type)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)

#         x = self.fc(x)

#         return F.sigmoid(x)