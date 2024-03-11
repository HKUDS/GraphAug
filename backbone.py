import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from layer_new import * # for mixing hop

class GCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(GCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.edge_attr is not None:
            edge_attr = data.edge_attr
            x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr))
            x = F.relu(self.conv2(x=x, edge_index=edge_index, edge_weight=edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class myGCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        node_embeddings = x

        return node_embeddings, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        
        self.upper_layers = [SparseNGCNLayer(in_channels = self.feature_number, out_channels = self.args.layers_1[i-1], iterations = i, dropout_rate = self.args.dropout) for i in range(1, self.order_1+1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(in_channels = self.abstract_feature_number_1, out_channels = self.args.layers_2[i-1], iterations = i, dropout_rate = self.args.dropout) for i in range(1, self.order_2+1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number).cuda()

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss
            
    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        abstract_features_1 = torch.cat([self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        # print("abstract_features_1:", abstract_features_1.size())
        abstract_features_2 = torch.cat([self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)], dim=1)
        # print("abstract_features_2:", abstract_features_2.size())
        node_emb = self.fully_connected(abstract_features_2)
        predictions = torch.nn.functional.log_softmax(node_emb, dim=1).cuda()
        # print("predictions:", node_emb.size(), predictions.size())
        # println()
        return node_emb, predictions




        