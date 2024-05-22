import torch
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.data import Data
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F

class SAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels: int, aggr = "mean", normalize: bool = False, root_weight: bool = True, project: bool = False, bias: bool = True, edge_dim=None, **kwargs):
        super().__init__(in_channels, out_channels, aggr, normalize, root_weight, project, bias, **kwargs)
        self.lin_edge = None
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, in_channels, bias=False,
                                   weight_initializer='glorot')
            self.lin_edge.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None):

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        if self.lin_edge is not None and edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
    
    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j