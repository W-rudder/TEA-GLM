import torch as t
from torch import nn
import torch.nn.functional as F
import math
from torch_geometric.utils import unbatch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from .layers import SAGEConv
from transformers.configuration_utils import PretrainedConfig
from torch_geometric.nn.conv import GATConv, GCNConv


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = t.zeros(q_len, d_model)
    position = t.arange(0, q_len).unsqueeze(1)
    div_term = t.exp(t.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = t.sin(position * div_term)
    pe[:, 1::2] = t.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def pos_encoding(pe, learn_pe, nvar, d_model):
    # Positional encoding
    if pe == None:
        W_pos = t.empty((nvar, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = t.empty((nvar, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = t.empty((nvar, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = t.zeros((nvar, 1))
        t.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = t.zeros((nvar, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos': W_pos = PositionalEncoding(nvar, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class Projector(nn.Module):
    def __init__(self,args, llama_embed):
        super(Projector, self).__init__()

        self.embed_tokens=llama_embed
        self.graph_projector = nn.Linear(args.gnn_input, self.embed_tokens.shape[1])

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, node_features, edge_index, mapping):
        assert type(node_features) is list
        node_features = t.cat(node_features, dim=0)
        assert mapping[-2] == node_features.shape[0]-1
        assert len(mapping) == node_features.shape[0]+len(edge_index)
        node_embedding = self.graph_projector(node_features)

        inputs_embeds = self.embed_tokens[input_ids]
        inputs_embeds[is_node] = node_embedding[mapping]

        return inputs_embeds


class LlamaEmbedding(nn.Module):
    def __init__(self,args, llama_embed):
        super(LlamaEmbedding, self).__init__()
        self.embed_tokens=t.nn.Embedding.from_pretrained(llama_embed)
    

    def get_input_embeddings(self):
        return self.embed_tokens


    def forward(self, input_ids, is_node, node_features, edge_index, mapping):
        embed_tokens = self.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        return inputs_embeds


class SoftEmbedding(nn.Module):
    def __init__(self, args):
        super(SoftEmbedding, self).__init__()
        self.args = args
        self.n_tokens = args.num_token
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding())
            
    def initialize_embedding(self):
        return t.FloatTensor(self.n_tokens, 4096).uniform_(-0.5, 0.5)
            
    def forward(self, input_ids):
        learned_embedding = self.learned_embedding.repeat(input_ids.shape[0], 1)
        return learned_embedding


class GraphEncoder(nn.Module):    # first_model, i.e. simple MLP for dimension transformation of OGB/ GIANT node embedding + freezed Llama-7b word embeddings.
    def __init__(self, args, llama_embed):
        super(GraphEncoder, self).__init__()
        self.args = args
        if args.gnn_type == 'GraphTransformer':
            self.GT = GraphTransformer(args)
        elif args.gnn_type == 'GraphSAGE':
            self.GT = GraphSAGE(args)
        elif args.gnn_type == 'SoftPrompt':
            self.GT = GraphSAGE(args)

        self.embed_tokens = llama_embed   # freezed Llama-7b word embeddings.
        self.embed_dim = llama_embed.shape[1]
        # self.graph_projector = nn.Sequential(
        #     nn.Linear(args.gnn_output, self.args.neck),
        #     nn.ReLU(),
        #     nn.Linear(self.args.neck, self.args.num_token * self.embed_dim),
        #     )
        if args.gnn_type == 'SoftPrompt':
            self.graph_projector = SoftEmbedding(args)
        else:
            self.graph_projector = nn.Sequential(
                nn.Linear(args.gnn_output, self.args.num_token * self.embed_dim),
                )

        # self.vocab_size = self.embed_tokens.shape[0]
        # self.num_tp = 1000
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tp)
        # self.graph_projector = nn.Sequential(
        #     nn.Linear(args.gnn_output, self.args.neck),
        #     nn.ReLU(),
        #     nn.Linear(self.args.neck, self.embed_dim),
        #     )
        # self.graph_projector = nn.Sequential(
        #     nn.Linear(args.gnn_output, self.args.neck),
        #     nn.ELU(),
        #     nn.Linear(self.args.neck, self.embed_dim),
        #     nn.ELU(),
        #     )
        # self.graph_projector = nn.Sequential(
        #     nn.Linear(args.gnn_output, self.embed_dim),
        #     )
        # self.attention_layer = Attention_layer(self.embed_dim, self.args.num_token, d_llm=self.embed_dim)

        # self.attention_layer = Attention_layer(self.embed_dim, 8, d_llm=self.embed_dim)
        # self.graph_projector = nn.Sequential(
        #     nn.Linear(args.gnn_output, self.args.neck),
        #     nn.ReLU(),
        #     nn.Linear(self.args.neck, self.args.num_token * self.embed_dim),
        #     )
        
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, graph, use_llm=False):
        batch_size = input_ids.shape[0]

        # old method
        if self.args.gnn_type == 'SoftPrompt':
            node_embeddings = self.graph_projector(input_ids)
        else:
            node_embedding = self.graph_projector(self.GT(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.lp))
            node_embeddings = node_embedding.view(-1, self.embed_dim) # [bs * 5, dim]
            if self.args.mask_token_list is not None:
                mask_list = [int(token) for token in self.args.mask_token_list.split(',')]
                mask = t.tensor([False if token_id in mask_list else True for token_id in range(self.args.num_token)])
                mask = mask.repeat(batch_size)
                node_embeddings = node_embeddings[mask]
                assert node_embeddings.shape[0] == batch_size * (self.args.num_token - len(mask_list))

        # new method
        # node_embedding = self.graph_projector(self.GT(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.lp))
        # source_embeddings = self.mapping_layer(self.embed_tokens.permute(1, 0)).permute(1, 0)
        # node_embeddings = self.attention_layer(node_embedding, source_embeddings, source_embeddings) # [bs * 5, dim]
        # new 2.0
        # node_embedding = self.GT(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.lp)
        # source_embeddings = self.mapping_layer(self.embed_tokens.permute(1, 0)).permute(1, 0)
        # node_embeddings = self.attention_layer(node_embedding, source_embeddings, source_embeddings) # [bs * 5, dim]
        # node_embeddings = self.graph_projector(node_embeddings).view(-1, self.embed_dim)

        inputs_embeds = self.embed_tokens[input_ids]
        if not use_llm:
            inputs_embeds[is_node] = node_embeddings

        return inputs_embeds # [bsz, seq, dim]

    @t.no_grad()
    def get_token_embeddings(self, input_ids, is_node, graph):
        batch_size = input_ids.shape[0]

        # old method
        node_embedding = self.graph_projector(self.GT(graph.x, graph.edge_index, graph.edge_attr, graph.batch, graph.lp))
        node_embedding = t.cat([node_embedding, graph.y], dim=-1)

        return node_embedding # [bsz, seq, dim]     

class GraphSageEncoder(nn.Module):    # first_model, i.e. simple MLP for dimension transformation of OGB/ GIANT node embedding + freezed Llama-7b word embeddings.
    def __init__(self, args, llama_embed):
        super(GraphSageEncoder, self).__init__()
        self.GT = GraphSAGE(args)

        self.embed_tokens=llama_embed   # freezed Llama-7b word embeddings.
        
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, input_ids, is_node, node_features, edge_index, mapping):
        assert type(node_features) is list
        node_embeddings = []
        for i in range(len(node_features)):
            node_embedding = self.GT(node_features[i], edge_index[i])[0] # [num_node, dim]
            node_embeddings.append(node_embedding)
            
        node_embeddings = t.cat(node_embeddings, dim=0)

        inputs_embeds = self.embed_tokens[input_ids]
        inputs_embeds[is_node] = node_embeddings
        # inputs_embeds[is_node] = (self.text_prototype[8] + self.text_prototype[9]).mean()

        return inputs_embeds # [bsz, seq, dim]


class GraphTransformer(nn.Module):
    def __init__(self, args):
        super(GraphTransformer, self).__init__()
        self.config = PretrainedConfig()
        self.args = args
        self.gtLayers = nn.Sequential(*[GTLayer(args) for i in range(args.gt_layers)])

        if self.args.if_pos: 
            self.W_pos = pos_encoding('zeros', True, 1, args.att_d_model)
                
        self.W_P = nn.Linear(args.gnn_input, args.att_d_model)
        self.dropout = nn.Dropout(0.1)
        self.inverW_P = nn.Linear(args.att_d_model, args.gnn_output)

    def forward(self, node_features, edge_index):
        # Adj: sp adj
        # x: bs * n * d_model * num_patch
        
        x = node_features.type_as(self.W_P.bias)
        
        # x, W_P_weight, W_P_bias= Mv2Samedevice([x, self.W_P.weight, self.W_P.bias])
        # self.W_P.weight = nn.Parameter(W_P_weight.to(x.dtype))
        # self.W_P.bias = nn.Parameter(W_P_bias.to(x.dtype))
        # print(self.W_P.bias.dtype, x.dtype)
        z = self.W_P(x)
        if self.args.if_pos: 
            embeds = self.dropout(z + self.W_pos) 
        else: 
            embeds = self.dropout(z) 
        for gt in self.gtLayers:
            embeds = gt(edge_index, embeds) # bs * num_patch * n * d_model
        # embeds, inverW_P_weight, inverW_P_bias = Mv2Samedevice([embeds, self.inverW_P.weight, self.inverW_P.bias])
        # self.inverW_P.weight = nn.Parameter(inverW_P_weight.to(embeds.dtype))
        # self.inverW_P.bias = nn.Parameter(inverW_P_bias.to(embeds.dtype))
        ret = self.inverW_P(embeds)
        return ret


class GTLayer(nn.Module):
    def __init__(self, args):
        super(GTLayer, self).__init__()
        self.qTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.kTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        self.vTrans = nn.Parameter(init(t.empty(args.att_d_model, args.att_d_model)))
        if args.att_norm: 
            self.norm = nn.LayerNorm(args.att_d_model, eps=1e-6)
        self.args = args
        
        
    
    def forward(self, edge_index, embeds):
        # Adj: adj
        # x: n * d_model
        rows, cols = edge_index
        nvar, _ = embeds.shape
        # print(rows)
        # print(cols)

        rowEmbeds = embeds[rows, :]
        colEmbeds = embeds[cols, :]
        evar, _ = rowEmbeds.shape

        qEmbeds = (rowEmbeds @ self.qTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([evar, self.args.head, self.args.att_d_model // self.args.head])
        
        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        
        tem = t.zeros([nvar, self.args.head]).to(expAtt.device, dtype=expAtt.dtype)
        # print(tem.device, expAtt.device, rows.device)
        rows = rows.to(expAtt.device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows, :]
        att = expAtt / (attNorm + 1e-8) # bleh
        
        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([evar, self.args.att_d_model])
        tem = t.zeros([nvar, self.args.att_d_model]).to(resEmbeds.device, dtype=resEmbeds.dtype)
        rows = rows.to(resEmbeds.device)
        tem = tem.to(resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds) # nd
        resEmbeds = resEmbeds + embeds
        if self.args.att_norm: 
            resEmbeds = self.norm(resEmbeds)

        return resEmbeds


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

        if args.conv_type == 'sage':
            gnn_conv = SAGEConv
        elif args.conv_type == "gat":
            gnn_conv = GATConv
        elif args.conv_type == 'gcn':
            gnn_conv = GCNConv

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
        elif args.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = t.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = t.nn.Linear(emb_dim, 1))
        elif args.graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)

    def forward(self, x, edge_index, edge_attr=None, batch=None, lp=None):
        for i, conv in enumerate(self.convs):
            # x = conv(x, edge_index, edge_attr=edge_attr)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        if self.args.graph_unsup:
            x = self.pool(x, batch)
        else:
            x = list(unbatch(x, batch))
            xs = []
            assert lp.shape[0] == len(x)
            for i in range(len(x)):
                if lp[i].data.item() == True:
                    xs.append(x[i][0] + x[i][1])
                    # xs.append(x[i][1])
                else:
                    xs.append(x[i][0])
            x = t.stack(xs, dim=0)

        return x

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class Attention_layer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(Attention_layer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys, d_llm)
        # self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B * self.n_heads, -1)
        # out = out.reshape(B, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = t.einsum("bhe,she->bhs", target_embedding, source_embedding)

        A = self.dropout(t.softmax(scale * scores, dim=-1))
        reprogramming_embedding = t.einsum("bhs,she->bhe", A, value_embedding)

        return reprogramming_embedding