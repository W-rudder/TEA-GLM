import torch
import math
from torch_geometric.sampler import NeighborSampler, NegativeSampling
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.loader import LinkNeighborLoader

class NodeNegativeSampler(NeighborSampler):
    def __init__(
            self, 
            data, 
            num_neighbors, 
            subgraph_type = 'directional', 
            replace: bool = False, 
            disjoint: bool = False, 
            temporal_strategy: str = 'uniform', 
            time_attr=None, 
            weight_attr=None, 
            is_sorted: bool = False, 
            share_memory: bool = False, 
            directed: bool = True,
            neg_ratio = 1.0):
        super().__init__(data, num_neighbors, subgraph_type, replace, disjoint, temporal_strategy, time_attr, weight_attr, is_sorted, share_memory, directed)
        self.neg_sampler = NegativeSampling('triplet', neg_ratio) if neg_ratio > 0 else None

    def sample_negative_from_nodes(self, inputs):
        out = node_negative_sample(inputs, self._sample, self.num_nodes, neg_sampling=self.neg_sampler)
        if self.subgraph_type == SubgraphType.bidirectional:
            out = out.to_bidirectional()
        return out
    
def node_negative_sample(inputs, sample_fn, num_nodes, neg_sampling=None):
    r"""Performs sampling from a :class:`NodeSamplerInput`, leveraging a
    sampling function that accepts a seed and (optionally) a seed time as
    input. Returns the output of this sampling procedure."""
    node_label = None
    if inputs.input_type is not None:  # Heterogeneous sampling:
        seed = {inputs.input_type: inputs.node}
        seed_time = None
        if inputs.time is not None:
            seed_time = {inputs.input_type: inputs.time}
    else:  # Homogeneous sampling:
        if neg_sampling is not None:
            num_neg = math.ceil(inputs.node.numel() * neg_sampling.amount)
            # TODO: Do not sample false negatives.
            # TODO: Remove edges between pos and neg
            # QUESTION: sample for pos and neg twice
            neg = neg_sampling.sample(num_neg, num_nodes)
            raw_nodes = torch.cat([inputs.node, neg])
            seed, inverse_seed = torch.cat([inputs.node, neg]).unique(return_inverse=True)

            node_label = torch.ones(inputs.node.numel())
            size = (num_neg, )
            node_neg_label = node_label.new_zeros(size)
            node_label = torch.cat([node_label, node_neg_label])
        else:
            raw_nodes = inputs.node
            seed, inverse_seed = inputs.node.unique(return_inverse=True)
            node_label = torch.ones(inputs.node.numel())

        seed_time = inputs.time

    out = sample_fn(seed, seed_time)
    out.metadata = (inputs.input_id, inputs.time, inverse_seed, node_label, raw_nodes)

    return out