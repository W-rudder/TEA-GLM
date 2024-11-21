import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from tqdm import tqdm
from layers import SAGEConv
from torch_geometric.nn.conv import GATConv, GCNConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, num_proj_hidden, activation, dropout, graph_pooling='sum', edge_dim=None, gnn_type='sage'):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = hidden_channels
        self.n_classes = out_channels
        self.convs = torch.nn.ModuleList()
        if gnn_type == 'sage':
            gnn_conv = SAGEConv
        elif gnn_type == "gat":
            gnn_conv = GATConv
        elif gnn_type == 'gcn':
            gnn_conv = GCNConv

        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = torch.nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, out_channels)

        self.dropout = dropout
        self.activation = activation

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is not None:
            x = self.pool(x, batch)

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
