from torch_geometric.loader.node_loader import NodeLoader
import torch
from copy import deepcopy
from sampler import NodeNegativeSampler
from torch_geometric.sampler import (
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.loader.utils import filter_data
from torch_geometric.utils import dropout_edge

class NodeNegativeLoader(NodeLoader):
    def __init__(
        self,
        data,
        num_neighbors,
        input_nodes = None,
        input_time = None,
        replace: bool = False,
        subgraph_type = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr = None,
        weight_attr = None,
        transform = None,
        transform_sampler_output = None,
        is_sorted: bool = False,
        filter_per_worker = None,
        neighbor_sampler = None,
        directed: bool = True,  # Deprecated.
        neg_ratio=1.0,
        mask_feat_ratio_1=0.3,
        mask_feat_ratio_2=0.2,
        drop_edge_ratio_1=0.1,
        drop_edge_ratio_2=0.4,
        mask1=None,
        mask2=None,
        **kwargs,
    ):
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        if neighbor_sampler is None:
            neighbor_sampler = NodeNegativeSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
                neg_ratio=neg_ratio,
            )

        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
        if mask1 is not None:
            self.mask1 = mask1
            self.mask2 = mask2
        else:
            self.mask1 = torch.rand(data.x.size(1)) < mask_feat_ratio_1
            self.mask2 = torch.rand(data.x.size(1)) < mask_feat_ratio_2

        self.mask_feat_ratio_1 = mask_feat_ratio_1
        self.mask_feat_ratio_2 = mask_feat_ratio_2
        self.drop_edge_1 = drop_edge_ratio_1
        self.drop_edge_2 = drop_edge_ratio_2


    def collate_fn(self, index):
        r"""Samples a subgraph from a batch of input nodes."""
        input_data: NodeSamplerInput = self.input_data[index]

        out = self.node_sampler.sample_negative_from_nodes(input_data)

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out
    
    def filter_fn(
        self,
        out,
    ):
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.node_sampler.edge_permutation)

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                data.e_id = perm[edge] if perm is not None else edge

            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)
            data.node_label_index = out.metadata[2]
            data.node_label = out.metadata[3]
            data.raw_nodes = out.metadata[4]

        view_1 = deepcopy(data)
        view_2 = deepcopy(data)

        # mask1 = torch.rand(data.x.size(1)) < self.mask_feat_ratio_1
        # mask2 = torch.rand(data.x.size(1)) < self.mask_feat_ratio_2

        # view_1.x[:, mask1] = 0
        # view_2.x[:, mask2] = 0

        # mask1 = torch.rand(data.x.shape[0], data.x.size(1)) < self.mask_feat_ratio_1
        # mask2 = torch.rand(data.x.shape[0], data.x.size(1)) < self.mask_feat_ratio_2

        # view_1.x[mask1] = 0
        # view_2.x[mask2] = 0

        view_1.x[:, self.mask1] = 0
        view_2.x[:, self.mask2] = 0

        view_1.edge_index, _ = dropout_edge(view_1.edge_index, self.drop_edge_1)
        view_2.edge_index, _ = dropout_edge(view_2.edge_index, self.drop_edge_2)

        return data, view_1, view_2