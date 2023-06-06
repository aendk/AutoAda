import dgl
import numpy as np
import torch
import os
import os.path as osp
from os.path import exists
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import save_graphs, load_graphs
from cli import parse_arguments


class PartitionedOGBDataset:
    def __init__(self, name, parts, root="dataset"):
        self.name = name
        self.parts = parts
        self.root = root
        filename = os.path.join(root, name.replace('-', '_'), f"graph.bin")
        graphs, self.idx_split = load_graphs(filename)
        self.graph = graphs[0]
        self.graph = dgl.add_reverse_edges(self.graph)  # Add reverse edges for unidirectional dataset.

    def num_features(self):
        return self.graph.num_nodes()

    def num_classes(self):
        labels = self.graph.ndata['label']
        return int(labels[labels.isfinite()].max().item()) + 1

    def load_split(self, part_id):
        filename = os.path.join(self.root, self.name.replace('-', '_'), f"{self.parts}parts", f"splits{part_id}.bin")
        return torch.load(filename)

    def prepareDataset(name, parts, root="dataset"):
        dataset = DglNodePropPredDataset(name, root)
        graph, labels = dataset[0]
        graph.ndata.clear()
        graph.ndata["label"] = labels[:, 0]

        idx_split = dataset.get_idx_split()
        train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
        train_mask[idx_split['train']] = True

        filename = os.path.join(root, name.replace('-', '_'), f"graph.bin")
        save_graphs(filename, graph, idx_split)

        if parts > 1:
            part_dir = os.path.join(root, name.replace('-', '_'), f"{parts}parts")
            os.makedirs(part_dir, exist_ok=True)
            partition = dgl.metis_partition(graph, parts, extra_cached_hops=0, reshuffle=False,
                                            balance_ntypes=train_mask, balance_edges=True)
            for i in range(parts):
                train_ids = torch.tensor(np.intersect1d(partition[i].ndata["_ID"], idx_split['train']))
                filename = os.path.join(part_dir, f"splits{i}.bin")
                torch.save(train_ids, filename)
