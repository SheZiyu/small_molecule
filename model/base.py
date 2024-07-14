import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
# import torch_cluster
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

# from small_sys_gnn.data.data import *
from prepocessing.preprocessing import parse_toml_file
from prepocessing.data import generate_train_dataset, data_augmentation

class MLP(nn.Module):
    '''
    A Multi-layer Perceptron (MLP) class implementing a simple feedforward neural network with
    customizable layer sizes and activation functions.

    Attributes:
        hidden_size_list (list of int): A list specifying the sizes of the input layer, any number of hidden layers,
                          and the output layer. For instance, [64, 128, 128, 10] would represent an
                          input layer with 64 neurons, two hidden layers with 128 neurons each, and
                          an output layer with 10 neurons.
        act (torch.nn.Module, optional): The activation function to be used for all hidden layers.
                                         Default is None, which means no activation function is applied.
                                         Example: torch.nn.ReLU().
        mlp (torch.nn.Sequential): The sequential container of linear layers and activation functions
                                   (if any) that constitute the MLP.

    Methods:
        forward(x): Defines the forward pass of the MLP.
        __repr__(): Returns a string representation of the MLP object.
    '''

    def __init__(self, hidden_size_list, cond_size_list=None, act=nn.ReLU(), is_cond=True, is_residual=False, drop=0.1):
        '''
        Initializes the MLP with specified layer sizes and activation function.

        Args:
            hidden_size_list (list of int): Sizes of the input, hidden, and output layers.
            act (torch.nn.Module, optional): Activation function for the hidden layers.
                                              Default is None.
            isresidual (bool, optional):
            drop (float)
        '''
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.cond_size_list = cond_size_list
        self.act = act
        self.is_cond = is_cond
        self.is_residual = is_residual
        self.drop = drop
        dropout = nn.Dropout(drop)

        layers = []
        for i in range(len(hidden_size_list) - 1):
            layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            if act is not None and i < len(hidden_size_list) - 2:
                layers.append(act)
                layers.append(dropout)
        self.mlp = nn.Sequential(*layers)

        if is_cond:
            cond_layers = []
            for i in range(len(cond_size_list) - 1):
                cond_layers.append(nn.Linear(cond_size_list[i], cond_size_list[i + 1]))
                if act is not None and i < len(cond_size_list) - 2:
                    cond_layers.append(act)
                    cond_layers.append(dropout)
            self.cond_mlp = nn.Sequential(*cond_layers)
            self.out = nn.Linear(hidden_size_list[-1] + cond_size_list[-1], hidden_size_list[-1])

        # self.apply(self._init_weights)

    def forward(self, x, cond=None):
        '''
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: The output tensor of the MLP after the forward pass.
        '''
        x = self.mlp(x) + x if self.is_residual else self.mlp(x)
        if self.is_cond:
            cond = self.cond_mlp(cond) + cond if self.is_residual else self.cond_mlp(cond)
            x = torch.cat([x, cond], dim=-1)
            x = self.out(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()

    def __repr__(self):
        '''
        Returns a string representation of the MLP object.

        Returns:
            str: String representation of the MLP.
        '''
        return '{}(hidden_size_list={}, cond_size_list={}, act={}, iscond={}, noresidual={}, drop={})'.format(
            self.__class__.__name__,
            self.hidden_size_list,
            self.cond_size_list,
            self.act,
            self.is_cond,
            self.is_residual,
            self.drop)

def aggregate(edge_index, edge_attr, node_attr, reduce='sum', is_cat=False, is_residual=False):
    '''
    :param edge_index: adjacency matrix of shape (2, num_edges)
    :param edge_attr: edge features of shape (num_edges, edge_dim)
    :param node_attr: node features of shape (num_nodes, node_dim)
    :param reduce: aggregation method to reduce, sum or mean
    :param cat: cutoff method to reduce, concatenate
    :param residual: cutoff method to reduce, sum
    :return: result of shape (num_nodes, edge_dim + node_dim) or (num_nodes, edge_dim)
    '''
    j = edge_index[1]
    result = scatter(edge_attr, index=j, dim=0, dim_size=node_attr.size(0), reduce=reduce)
    if is_cat:
        result = torch.cat([result, node_attr], dim=-1) # [bn, edge_dim + node_dim]
    if is_residual:
        result += node_attr # [bn, edge_dim]
    return result

class MessagePassingGNN(nn.Module):
    def __init__(self, vector_dim, scalars_dim, hidden_dim, output_dim, is_norm=True):
        '''
        :param vector_dim:
        :param scalars_dim:
        :param hidden_dim:
        :param output_dim:
        :param norm: normalization of vector features
        '''
        super(MessagePassingGNN, self).__init__()
        self.is_norm = is_norm
        self.mlp = MLP([vector_dim + scalars_dim, hidden_dim,
                        hidden_dim, output_dim],
                       [vector_dim, hidden_dim,
                        hidden_dim, output_dim])

    def forward(self, x, scalars, cond=None):
        '''
        :param x: vector features, e.g., positions, velocities, of shape (num_edges, vector_dim)
        :param scalars: scalar features, e.g., node features, edge features, of shape (num_edges, scalars_dim)
        :return: message scalar of shape (num_edges, output_dim)
        '''
        if self.is_norm:
            x = F.normalize(x, p=2, dim=-1)
            cond = F.normalize(cond, p=2, dim=-1)
        scalar = self.mlp(torch.cat([x, scalars], dim=-1), cond=cond)
        return scalar

class GNN_layer(nn.Module):
    '''
    GNN_layer is used to update node coordinates and node features, here we set edge_index and edge_attr as constants
    '''

    def __init__(self, node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim):
        '''
        :param node_dim:
        :param edge_dim:
        :param message_dim:
        :param hidden_dim:
        :param vector_dim:
        :param scalar_dim:
        '''
        super(GNN_layer, self).__init__()
        scalars_dim = node_dim * 2 + edge_dim + edge_type_dim
        self.edge_type_dim = edge_type_dim
        self.vector_dim = vector_dim
        self.scalar_dim = scalar_dim
        self.messageNN = MessagePassingGNN(vector_dim=vector_dim*2, scalars_dim=scalars_dim, output_dim=message_dim,
                                           hidden_dim=hidden_dim, is_norm=True)
        self.vectorNN = MLP([message_dim, hidden_dim,
                             hidden_dim, vector_dim],
                            [vector_dim*2, hidden_dim,
                             hidden_dim, vector_dim]
                            )
        self.scalarNN = MLP([message_dim, hidden_dim,
                             hidden_dim, scalar_dim],
                            is_cond=False
                            )
        self.nodeNN = MLP([scalar_dim, hidden_dim,
                           hidden_dim, node_dim],
                          is_cond=False
                          )

    def forward(self, edge_index, edge_type, edge_attr, x, h, cond):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :return: vector: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 node_attr: node features of shape (num_nodes, node_dim)
        '''
        row, col = edge_index[0], edge_index[1]
        xij = torch.cat([x[row], x[col]], dim=-1)
        # print(h[row], h[col], edge_attr)
        edge_type_one_hot = F.one_hot(edge_type, num_classes=self.edge_type_dim)
        hij = torch.cat([h[row], h[col], edge_attr, edge_type_one_hot], dim=-1)
        condij = torch.cat([cond[row], cond[col]], dim=-1)
        message = self.messageNN(xij, scalars=hij, cond=condij)  # [bm, message_dim]
        vector = self.vectorNN(message, cond=condij)  # [bm, vector_dim]
        vector = x + aggregate(edge_index, vector, x, reduce='sum', is_cat=False, is_residual=False)  # [bn, vector_dim]

        scalar = self.scalarNN(message)  # [bm, scalar_dim]
        node_attr = self.nodeNN(aggregate(edge_index, scalar, h, reduce='sum',  is_cat=False, is_residual=False))  # [bn, node_dim]
        node_attr += h
        return vector, node_attr, cond

class GNN(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim):
        '''
        :param num_layers:
        :param node_dim:
        :param edge_dim:
        :param message_dim:
        :param hidden_dim:
        :param vector_dim:
        :param scalar_dim:
        '''
        super(GNN, self).__init__()
        layers = list()
        gnn_layer = GNN_layer(node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        for i in range(num_layers):
            layers.append(gnn_layer)

        self.gnn = nn.ModuleList(layers)

    def forward(self, edge_index, edge_type, edge_attr, x, h, cond):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :return: x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 h: node features of shape (num_nodes, node_dim)
        '''
        for l in self.gnn:
            x, h, cond = l(edge_index, edge_type, edge_attr, x, h, cond)
        return x, h, cond

class MessagePassingEGNN(nn.Module):
    def __init__(self, num_vec, vector_dim, scalars_dim, hidden_dim, output_dim, is_norm=True, is_scalar=True):
        '''
        :param num_vec:
        :param scalars_dim:
        :param hidden_dim:
        :param output_dim:
        :param norm: normalization of vector features
        :param scalar: cutoff method to reduce, concatenate
        '''
        super(MessagePassingEGNN, self).__init__()
        self.num_vec = num_vec
        self.vector_dim = vector_dim
        self.scalars_dim = scalars_dim
        self.is_norm = is_norm

        if is_scalar:
            self.mlp = MLP([num_vec + scalars_dim, hidden_dim,
                            hidden_dim, output_dim],
                           [vector_dim, hidden_dim,
                            hidden_dim, output_dim]
                           )
        else:
            self.mlp = MLP([num_vec, hidden_dim,
                            hidden_dim, output_dim],
                           [vector_dim, hidden_dim,
                            hidden_dim, output_dim]
                           )

    def forward(self, x, scalars=None, cond=None):
        '''
        :param x: vector features, e.g., positions, velocities, of shape (num_edges, vector_dim)
        :param scalars: scalar features, e.g., node features, edge features, of shape (num_edges, scalars_dim)
        :return: message scalar of shape (num_edges, output_dim)
        '''
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        assert x.shape[-1] == self.num_vec

        x_t = x.transpose(-2, -1)
        scalar = torch.einsum('bij,bjk->bik', x_t, x)
        scalar = scalar.reshape([-1, self.num_vec * self.num_vec])
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)
            cond = F.normalize(cond, p=2, dim=-1)
        if scalars is not None:
            assert scalars.shape[-1] == self.scalars_dim
            scalar = torch.cat([scalar, scalars], dim=-1)
        scalar = self.mlp(scalar, cond=cond)
        return scalar

class EGNN_layer(nn.Module):
    '''
    EGNN_layer is used to update node coordinates and node features, here we set edge_index and edge_attr as constants
    '''
    def __init__(self, node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim):
        '''
        :param node_dim:
        :param edge_dim:
        :param message_dim:
        :param hidden_dim:
        :param vector_dim:
        :param scalar_dim:
        '''
        super(EGNN_layer, self).__init__()
        scalars_dim = node_dim * 2 + edge_dim + edge_type_dim
        self.edge_type_dim = edge_type_dim
        self.vector_dim = vector_dim
        self.scalar_dim = scalar_dim
        self.messageNN = MessagePassingEGNN(vector_dim=vector_dim*2, scalars_dim=scalars_dim, output_dim=message_dim, num_vec=1, hidden_dim=hidden_dim, is_norm=True, is_scalar=True)
        self.vectorNN = MLP([message_dim, hidden_dim,
                             hidden_dim, vector_dim],
                            [vector_dim*2, hidden_dim,
                             hidden_dim, vector_dim]
                            )
        self.scalarNN = MLP([message_dim, hidden_dim,
                             hidden_dim, scalar_dim],
                            is_cond=False
                            )
        self.nodeNN = MLP([scalar_dim, hidden_dim,
                           hidden_dim, node_dim],
                          is_cond=False
                          )

    def forward(self, edge_index, edge_type, edge_attr, x, h, cond):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :return: vector: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 node_attr: node features of shape (num_nodes, node_dim)
        '''
        row, col = edge_index[0], edge_index[1]
        xij = x[row] - x[col]
        edge_type_one_hot = F.one_hot(edge_type, num_classes=self.edge_type_dim)
        hij = torch.cat([h[row], h[col], edge_attr, edge_type_one_hot], dim=-1)
        condij = torch.cat([cond[row], cond[col]], dim=-1)
        message = self.messageNN(xij, scalars=hij, cond=condij) # [bm, message_dim]
        assert xij.shape[-1] == self.vector_dim
        vector = xij * self.vectorNN(message, cond=condij) # [bm, vector_dim]
        vector = x + aggregate(edge_index, vector, x, reduce='sum', is_cat=False, is_residual=False) # [bn, vector_dim]

        scalar = self.scalarNN(message) # [bm, scalar_dim]
        node_attr = self.nodeNN(aggregate(edge_index, scalar, h, reduce='sum', is_cat=False, is_residual=False)) # [bn, node_dim]
        node_attr += h
        return vector, node_attr, cond

class EGNN(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim):
        '''
        :param num_layers:
        :param node_dim:
        :param edge_dim:
        :param message_dim:
        :param hidden_dim:
        :param vector_dim:
        :param scalar_dim:
        '''
        super(EGNN, self).__init__()
        layers = list()
        egnn_layer = EGNN_layer(node_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        for i in range(num_layers):
            layers.append(egnn_layer)

        self.egnn = nn.ModuleList(layers)

    def forward(self, edge_index, edge_type, edge_attr, x, h, cond):
        '''
        :param edge_index: adjacency matrix of shape (2, num_edges)
        :param edge_attr: edge features of shape (num_edges, edge_dim)
        :param x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
        :param h: node features of shape (num_nodes, node_dim)
        :return: x: vector features, e.g., positions, velocities, of shape (num_nodes, vector_dim)
                 h: node features of shape (num_nodes, node_dim)
        '''
        for l in self.egnn:
            x, h, cond = l(edge_index, edge_type, edge_attr, x, h, cond)
        return x, h, cond

class FourierTimeEmbedding(nn.Module):
    def __init__(self, embed_dim, input_dim=1, sigma=1.0):
        '''
        :param embed_dim: time embedding dimension, the time embedding is constant
        :param input_dim:
        :param sigma:
        '''
        super(FourierTimeEmbedding, self).__init__()
        B = torch.nn.Parameter(torch.randn(input_dim, embed_dim // 2) * sigma, requires_grad=False)
        self.register_buffer('B', B)

    def forward(self, v):
        '''
        :param v: time indexes of shape (num_nodes, 1),
                  for per frame or per time step, the indexes are the same,
                  which means that the time embedding is the same
        :return: shape of (num_nodes, embed_dim)
        '''
        # device = next(self.parameters()).device  # Getting the device from the model's parameters
        # v = v.to(device)
        v_proj = 2 * torch.pi * v @ self.B
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)

class DynamicsGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, edge_type_dim, vector_dim,
                 model=GNN, num_layers=8,
                 message_dim=8, hidden_dim=512, scalar_dim=16
                 ):
        super(DynamicsGNN, self).__init__()
        self.time_embedding = FourierTimeEmbedding(embed_dim=hidden_dim, input_dim=1)
        self.model1 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.model2 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.model3 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        # Sanity check
        self.invariant_scalr = MLP([node_dim+hidden_dim, 1], is_cond=False)

    def forward(self, t, edge_index, edge_type, edge_attr, x, h, cond):
        t = self.time_embedding(t)
        # print(t.shape)
        # print(h.shape)
        h = torch.cat([h, t], dim=-1)
        x, h, cond = self.model1(edge_index, edge_type, edge_attr, x, h, cond)
        x, h, cond = self.model2(edge_index, edge_type, edge_attr, x, h, cond)
        x, h, cond = self.model3(edge_index, edge_type, edge_attr, x, h, cond)
        h = self.invariant_scalr(h)
        return x, h

    def reset_parameters(self):
        # Custom logic to reset or initialize parameters
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

class DynamicsEGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, edge_type_dim, vector_dim,
                 model=EGNN, num_layers=8,
                 message_dim=8, hidden_dim=512, scalar_dim=16
                 ):
        super(DynamicsEGNN, self).__init__()
        self.time_embedding = FourierTimeEmbedding(embed_dim=hidden_dim, input_dim=1)
        self.model1 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.model2 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        self.model3 = model(num_layers, node_dim+hidden_dim, edge_dim, edge_type_dim, message_dim, hidden_dim, vector_dim, scalar_dim)
        # Sanity check
        self.invariant_scalr = MLP([node_dim+hidden_dim, 1], is_cond=False)

    def forward(self, t, edge_index, edge_type, edge_attr, x, h, cond):
        t = self.time_embedding(t)
        h = torch.cat([h, t], dim=-1)
        x, h, cond = self.model1(edge_index, edge_type, edge_attr, x, h, cond)
        x, h, cond = self.model2(edge_index, edge_type, edge_attr, x, h, cond)
        x, h, cond = self.model3(edge_index, edge_type, edge_attr, x, h, cond)
        h = self.invariant_scalr(h)
        return x, h

    def reset_parameters(self):
        # Custom logic to reset or initialize parameters
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

if __name__ == '__main__':
    config = parse_toml_file('config.toml')
    directory_path = '/data2/ziyu_project/test'
    cutoff = config['cutoff']
    scale = config['scale']
    node_dim = config['node_dim']
    edge_dim = config['edge_dim']
    vector_dim = config['vector_dim']
    device = config['device']
    num_splits = config['num_splits']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    num_frames_to_process = config['num_frames_to_process']
    # ProteinAnalysis(directory_path, num_frames_to_process).preprocess_coordinate_onehot()
    TrajsDataset = TrajsDataset(
        False,
        directory_path,
        num_frames_to_process,
        cutoff=cutoff,
        scale=scale,
        file=None
    )
    train_loader = generate_train_dataset(TrajsDataset, batch_size, num_workers)

    dynamicsgnn = DynamicsGNN(node_dim, edge_dim, vector_dim).to(device)
    for batch in train_loader:
        batch = batch.to(device)
        batch = data_augmentation(batch)
        out_x, out_h = dynamicsgnn(t=batch.frame_idx,
                                   edge_index=batch.edge_index,
                                   edge_attr=batch.edge_encoding,
                                   x=batch.dd,
                                   h=batch.x,
                                   cond=batch.pos)
        print(out_x.shape, out_h.shape)
        break

    dynamicsegnn = DynamicsEGNN(node_dim, edge_dim, vector_dim).to(device)
    for batch in train_loader:
        batch = batch.to(device)
        batch = data_augmentation(batch)
        out_x, out_h = dynamicsegnn(t=batch.frame_idx,
                                    edge_index=batch.edge_index,
                                    edge_attr=batch.edge_encoding,
                                    x=batch.dd,
                                    h=batch.x,
                                    cond=batch.pos)
        print(out_x.shape, out_h.shape)
        break



