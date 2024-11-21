import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .layers import SAGEConv


class GraphEncoder(nn.Module):    # first_model, i.e. simple MLP for dimension transformation of OGB/ GIANT node embedding + freezed Llama-7b word embeddings.
    def __init__(self, args, llama_embed):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.GT = GraphSAGE(args)

        self.embed_tokens = llama_embed   # freezed Llama-7b word embeddings.
        self.embed_dim = llama_embed.shape[1]

        self.graph_projector = nn.Sequential(
            nn.Linear(args.gnn_output, self.args.num_token * self.embed_dim),
            )
        
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, graph):
        batch_size = input_ids.shape[0]

        node_embedding = self.graph_projector(self.GT(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.lp))
        node_embeddings = node_embedding.view(-1, self.embed_dim) # [bs * 5, dim]

        inputs_embeds = self.embed_tokens[input_ids]
        inputs_embeds[is_node] = node_embeddings

        return inputs_embeds # [bsz, seq, dim]


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = args.gnn_input
        n_layers = args.gt_layers
        hidden_channels = args.att_d_model
        out_channels = args.gnn_output
        edge_dim = args.edge_dim
        num_proj_hidden = out_channels

        gnn_conv = SAGEConv
        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        # non-linear layer for contrastive loss
        self.fc1 = nn.Linear(out_channels, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, out_channels)

        self.activation = F.relu

        #Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool

    def forward(self, x, edge_index, edge_attr=None, batch=None, lp=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        x = list(unbatch(x, batch))
        xs = []
        assert lp.shape[0] == len(x)
        for i in range(len(x)):
            if lp[i].data.item() == True:
                xs.append((x[i][0] + x[i][1]) / 2)
            else:
                xs.append(x[i][0])
        x = t.stack(xs, dim=0)

        return x

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)