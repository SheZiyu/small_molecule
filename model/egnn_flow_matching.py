import torch 
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

# env: equivariant_gflownet5
from math import pi as PI

from prepocessing.transforms import extend_to_radius_graph
from model.base import FourierTimeEmbedding


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=3,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2  # why *2?
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1  # what is edge coordis_nf?

        # 257
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                input_edge + edge_coords_nf + edges_in_d, hidden_nf
            ),  # own: check if the input is 64+64+1+1
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        # print("edge_model", radial, edge_attr)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat(
                [source, target, radial, edge_attr], dim=1
            )  # own: edge_attr==radial
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_feat, node_attr):
        # print("node_model", edge_feat)
        src, dst = edge_index
        agg = unsorted_segment_sum(edge_feat, src, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(
        self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
    ):
        # print("coord_model", coord_diff, radial, edge_feat)
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        # trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == "mean":
            if node_mask is not None:
                # raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(
                    node_mask[col], row, num_segments=coord.size(0)
                )
                agg = agg / (M - 1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        # print("update", coord, coord_diff,edge_feat, self.coord_mlp(edge_feat), self.coords_range, agg, self.tanh)
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        src, dst = edge_index
        radial, coord_diff = self.coord2radial(
            edge_index, coord
        )  # own: radial is the distance squared and coord_diff is the normalized difference vector

        edge_feat = self.edge_model(
            h[src], h[dst], radial, edge_attr, edge_mask
        )  # in the egnn, this is (3)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )  # in the egnn, this is (4)

        h, agg = self.node_model(
            h, edge_index, edge_feat, node_attr
        )  # in the egnn, this is (6)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[dst], u, batch)  # GCN
        # print("h", h)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_feat

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(
            1
        )  # own: this is just the distance squared

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)

        return radial, coord_diff


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        # hidden_nf=128,
        hidden_nf=64,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=5,
        recurrent=True,
        attention=True,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        """EGNN model

        Args:
            in_node_nf : node feature dimension
            in_edge_nf : edge feature dimension
            hidden_nf: Hidden dimension. Defaults to 64.
            device (str, optional): device to compute on. Defaults to "cpu".
            act_fn : Activation function. Defaults to nn.SiLU().
            n_layers (int, optional): Number of E_GCL layers. Defaults to 5.
            recurrent (bool, optional): If True, the node location will be added to the output of the node model.  Defaults to True.
            attention (bool, optional): If True, apply attention. Defaults to True.
            norm_diff (bool, optional): Seems not to be used?. Defaults to True.
            out_node_nf: In the beginning and in the end an embedding is applied to the node features. Defaults to None.
            tanh (bool, optional): _description_. Defaults to False.
            coords_range (int, optional): _description_. Defaults to 15.
            agg (str, optional): _description_. Defaults to "sum".
        """
        # in the original code in_node_nf was the one hot encoding of the atom type
        # in_edge_nf was set to one and as input just used the distance squared
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

        self.to(self.device)

    def forward(
        self,
        h_ininitial,
        x_mean_free,
        edges,
        node2graph,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
        extend_radius=True,
    ):
        h = self.embedding(h_ininitial)
        if extend_radius:
            new_edge_index, new_edge_type = extend_to_radius_graph(
                x_mean_free,
                edges,
                edge_attr,
                2.0,
                node2graph,
                unspecified_type_number=0,
            )
            # edges = create_distance_edges_within_graphs(x, batch2graph, 10.0)
        edge_type_one_hot = torch.nn.functional.one_hot(new_edge_type, num_classes=3)
        for i in range(0, self.n_layers):
            h, x_mean_free, _ = self._modules["gcl_%d" % i](
                h,
                new_edge_index,
                x_mean_free,
                edge_attr=edge_type_one_hot,
                node_mask=node_mask,
                edge_mask=edge_mask,  # what is the purpose of edge_mask?
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x_mean_free
    
    
def create_distance_edges_within_graphs(coords, node2graph, cutoff_radius):
    """
    Calculates distances between points withing a graph and returns edges

    Args:
        coords (torch.Tensor): A tensor of coordinates with shape (num_points, num_dims),
                               where num_points is the number of points and num_dims is the dimensionality of each point.
        node2graph: tensor specifying to which graph a node belongs
        cutoff_radius (float): The radius within which edges are considered.


    """
    num_points = coords.shape[0]
    # Create a square matrix of graph ids
    graph_ids = node2graph.unsqueeze(1).repeat(1, num_points)
    # Create masks for identifying nodes belonging to the same graph
    same_graph_mask = graph_ids == graph_ids.T

    # Calculate relative coordinates (vector differences) between all points
    rel_coords = coords[:, None] - coords[None, :]
    # Calculate distances between all points
    dis = torch.norm(rel_coords, dim=-1)

    # Apply cutoff radius and same graph mask
    edge_mask = (dis < cutoff_radius) & same_graph_mask
    edge_indices = torch.nonzero(edge_mask, as_tuple=False).T

    return edge_indices


def subtract_means(stacked_points, node2graph):
    """Center the stacked_points by subtracting the corresponding means. node2graph specifies wich points belong to the same graph.

    Args:
        stacked_points: tensor of atom points
        node2graph: tensor of indices specifying to which graph a node belongs

    Returns:
        centered_points: centered points
    """
    means = scatter_mean(
        stacked_points, node2graph, dim=0, dim_size=node2graph.max() + 1
    )
    centered_points = stacked_points - means.index_select(0, node2graph)
    return centered_points


class DynamicsEGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf=64, model=EGNN):
        super(DynamicsEGNN, self).__init__()
        self.time_embedding = FourierTimeEmbedding(embed_dim=hidden_nf, input_dim=1)
        self.model1 = model(in_node_nf+hidden_nf, in_edge_nf, hidden_nf)
        self.model2 = model(in_node_nf, in_edge_nf, hidden_nf)
        self.model3 = model(in_node_nf, in_edge_nf, hidden_nf)
    

    def forward(self, t, h, x, edge_index, node2graph, edge_attr=None):
        t = self.time_embedding(t)
        h = torch.cat([t, h], dim=-1)
        h, x  = self.model1(h, x, edge_index, node2graph, edge_attr)
        # print(h.shape) 
        # print(x.shape)
        return x, h

    def reset_parameters(self):
        # Custom logic to reset or initialize parameters
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()