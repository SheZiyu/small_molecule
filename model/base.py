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
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

from prepocessing.data_extend import *
from model.egnn_flow_matching import extend_to_radius_graph


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

    def __init__(self, hidden_size_list, act=None, drop=0.1):
        '''
        Initializes the MLP with specified layer sizes and activation function.

        Args:
            hidden_size_list (list of int): Sizes of the input, hidden, and output layers.
            act (torch.nn.Module, optional): Activation function for the hidden layers.
                                              Default is None.
            noresidual (bool, optional):
            drop (float)
        '''
        super().__init__()
        self.hidden_size_list = hidden_size_list
        self.act = act
        self.drop = drop
        dropout = nn.Dropout(drop)

        layers = []
        for i in range(len(hidden_size_list) - 1):
            layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            if act is not None and i < len(hidden_size_list) - 2:
                layers.append(act)
                # layers.append(dropout)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: The output tensor of the MLP after the forward pass.
        '''
        return self.mlp(x)

    def __repr__(self):
        '''
        Returns a string representation of the MLP object.

        Returns:
            str: String representation of the MLP.
        '''
        return '{}(hidden_size_list={}, act={}, drop={})'.format(
            self.__class__.__name__,
            self.hidden_size_list,
            self.act,
            self.drop)


class EdgeProcessor(nn.Module):
    def __init__(self, hidden_size_list_h, hidden_size_list_x):
        super().__init__()
        self.edge_mlp_h = nn.Sequential(MLP(hidden_size_list=hidden_size_list_h, act=nn.SiLU()), nn.LayerNorm(hidden_size_list_h[-1]))
        self.edge_mlp_x = nn.Sequential(MLP(hidden_size_list=hidden_size_list_x, act=nn.SiLU()),
                                        nn.LayerNorm(hidden_size_list_x[-1]))
        self.embedding_out = nn.Linear(2 * hidden_size_list_h[-1], hidden_size_list_h[-1])

    def forward(self, h_i, h_j, x_i, x_j, edge_type):
        out_h = self.edge_mlp_h(
            torch.cat([h_i, h_j, edge_type], dim=-1)
        )
        disp_vec = x_j - x_i
        disp_len = torch.linalg.norm(disp_vec, dim=-1, keepdim=True)
        out_x = self.edge_mlp_x(
            torch.cat([disp_vec, disp_len], dim=-1)
        )
        # out_x = self.edge_mlp_x(disp_vec)
        out = self.embedding_out(
            torch.cat([out_h, out_x], dim=-1))
        return out

class NodeProcessor(nn.Module):
    def __init__(self, hidden_size_list):
        super().__init__()
        self.node_mlp = nn.Sequential(MLP(hidden_size_list=hidden_size_list, act=nn.SiLU()),
                                      nn.LayerNorm(hidden_size_list[-1]))

    def forward(self, node_attr, edge_index, edge_attr):
        j = edge_index[1]
        out = scatter(edge_attr, index=j, dim=0, dim_size=node_attr.size(0))
        out = torch.cat([node_attr, out], dim=-1)
        out = self.node_mlp(out)
        out += node_attr
        return out

class PosProcessor(nn.Module):
    def __init__(self, hidden_size_list):
        super().__init__()
        self.pos_mlp = nn.Sequential(MLP(hidden_size_list=hidden_size_list, act=nn.SiLU()), nn.LayerNorm(hidden_size_list[-1]))

    def forward(self, x, edge_index, edge_attr):
        j = edge_index[1]
        out = scatter(edge_attr, index=j, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=-1)
        out = self.pos_mlp(out)
        out += x
        return out

class EdgeNodePosConv(nn.Module):
    def __init__(self, node_dim, pos_dim, hidden_dim, edge_type_dim):
        super().__init__()
        self.node_dim = node_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.edge_type_dim = edge_type_dim
        self.node_processor = NodeProcessor([node_dim + edge_type_dim, hidden_dim, hidden_dim, node_dim])
        self.edge_processor = EdgeProcessor([node_dim * 2 + edge_type_dim, hidden_dim, hidden_dim, hidden_dim],
                                            [pos_dim * 1 + 1, hidden_dim, hidden_dim, hidden_dim])
        self.pos_processor = PosProcessor([pos_dim + hidden_dim, hidden_dim, hidden_dim, pos_dim])

    def forward(self, node_h, x, edge_index, edge_type_h):
        i = edge_index[0]
        j = edge_index[1]

        node_h = self.node_processor(node_h, edge_index, edge_type_h)
        edge_h = self.edge_processor(node_h[i], node_h[j], x[i], x[j], edge_type_h)
        # print(x.shape)
        x = self.pos_processor(x, edge_index, edge_h)
        return edge_h, node_h, x

    def __repr__(self):
        return f'{self.__class__.__name__}(node_dim={self.node_dim}, pos_dim={self.pos_dim})'

class EdgeNodePosTransformer(nn.Module):
    def __init__(self, num_convs, node_dim, pos_dim, hidden_dim, edge_type_dim):
        super().__init__()
        self.num_convs = num_convs
        self.node_dim = node_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.edge_type_dim = edge_type_dim
        self.convs = nn.ModuleList([copy.deepcopy(
            EdgeNodePosConv(node_dim, pos_dim, hidden_dim, edge_type_dim)) for _ in range(num_convs)])

    # def forward(self, node_h, x, edge_index, edge_type_h) :
    #     for conv in self.convs:
    #         edge_h, node_h, x = conv(node_h, x, edge_index, edge_type_h)
    #     return edge_h, node_h, x

    def forward(
            self,
            h_initial,
            original_nodes,
            perturbed_nodes,
            edge_index,
            edge_type,
            node2graph
    ):
        original_features = torch.cat([h_initial, 56*torch.ones([h_initial.size(0), 1], device=h_initial.device)], dim=1)
        perturbed_features = torch.cat([h_initial, -56*torch.ones([h_initial.size(0), 1], device=h_initial.device)], dim=1)
        h = torch.cat([original_features, perturbed_features], dim=0)

        new_edge_index, new_edge_type = extend_to_radius_graph(
            original_nodes,
            perturbed_nodes,
            edge_index,
            edge_type,
            56,
            node2graph
        )

        edge_type_one_hot = torch.nn.functional.one_hot(new_edge_type, num_classes=self.edge_type_dim)
        x = torch.cat([original_nodes, perturbed_nodes], dim=0)

        for conv in self.convs:
            edge_h, h, x = conv(h, x, new_edge_index, edge_type_one_hot)
            x = torch.cat([original_nodes, x[original_nodes.shape[0]:, :]], dim=0)

        return h[original_nodes.shape[0]:, :], x[original_nodes.shape[0]:, :]


class DynamicsGNN(nn.Module):
    def __init__(self, time_encoder, transformer):
        super().__init__()
        self.time_encoder = time_encoder
        self.transformer = transformer

    def forward(self,
                t,
                h,
                original_nodes,
                perturbed_nodes,
                edge_index,
                edge_type,
                node2graph):
        t = self.time_encoder(t)
        h = torch.cat([t, h], dim=-1)
        h, x = self.transformer(
            h,
            original_nodes,
            perturbed_nodes,
            edge_index,
            edge_type,
            node2graph)
        return h, x


time_encoder = FourierTimeEmbedding(embed_dim=512)
edge_node_pos_transformer = EdgeNodePosTransformer(num_convs=5, node_dim=530, pos_dim=3, hidden_dim=512, edge_type_dim=4)
dynamicsGNN = DynamicsGNN(time_encoder=time_encoder, transformer=edge_node_pos_transformer)