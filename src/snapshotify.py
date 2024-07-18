import random
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm


def add_negative_edges(edge_index):
    all_nodes = list(range(edge_index.max().item() + 1))
    num_edges = edge_index.size(1)
    negative_edges = []
    make_sure_not_positive = False
    if make_sure_not_positive:
        while len(negative_edges) < num_edges:
            u, v = random.sample(all_nodes, 2)
            negative_edges.append([u, v])
        negative_edge_index = torch.tensor(negative_edges).t().contiguous()
    else:
        negative_edge_index = torch.from_numpy(np.random.choice(all_nodes, (2, num_edges))).contiguous()

    negative_labels = torch.zeros(negative_edge_index.size(1))
    positive_edges = torch.ones(negative_edge_index.size(1))

    all_edge_index = torch.cat([edge_index, negative_edge_index], dim=1)
    all_labels = torch.cat([positive_edges, negative_labels])
    return all_edge_index, all_labels


class TGBDataset:
    def __init__(self, name, time_interval, num_features):
        if 'tgbl' in name:
            self.loader = PyGLinkPropPredDataset(name=name, root="datasets")
        elif 'tgbn' in name:
            self.loader = PyGNodePropPredDataset(name=name, root="datasets")
        else:
            raise ValueError('Unsupported dataset name')

        self.metric = self.loader.eval_metric
        self.evaluator = Evaluator(name)
        self.neg_sampler = self.loader.negative_sampler
        self.num_features = num_features
        self.time_interval = time_interval

        train_mask = self.loader.train_mask
        val_mask = self.loader.val_mask
        test_mask = self.loader.test_mask
        data = self.loader.get_TemporalData()

        train_data = data[train_mask]
        val_data = data[val_mask]
        test_data = data[test_mask]
        self.max_node_till_now = 0

        self.train_loader = TemporalDataLoader(train_data, batch_size=time_interval)
        self.val_loader = TemporalDataLoader(val_data, batch_size=time_interval)
        self.test_loader = TemporalDataLoader(test_data, batch_size=time_interval)
        self.random_node_pe = defaultdict(lambda: torch.randn(self.num_features))

    def create_snapshots(self):
        train_snapshot = self.loader_to_snapshots(self.train_loader, split='train')
        self.loader.load_val_ns()
        val_snapshot = self.loader_to_snapshots(self.val_loader, split='val')
        self.loader.load_test_ns()
        test_snapshot = self.loader_to_snapshots(self.test_loader, split='test')

        return train_snapshot, val_snapshot, test_snapshot

    def loader_to_snapshots(self, loader, split='train'):
        snapshots = []
        for batch in tqdm(loader, desc=f'{split} loader to snapshots'):
            pos_src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            edge_index = torch.stack([pos_src, pos_dst], dim=0)
            if split == 'train':
                all_edge_index, all_labels = add_negative_edges(edge_index)
                all_id = torch.cat([torch.arange(len(pos_src)), torch.arange(len(pos_src))])
            elif split in ['test', 'val']:
                neg_batch_list = self.neg_sampler.query_batch(pos_src, pos_dst, t, split_mode=split)
                all_id = torch.arange(len(neg_batch_list))
                all_edge_index = edge_index.clone()  # Clone to preserve original edges
                all_labels = torch.zeros(all_edge_index.size(1), dtype=torch.float)  # Placeholder labels
                new_edges_list = []
                new_labels_list = []
                new_id_list = []
                for idx, neg_batch in enumerate(neg_batch_list):
                    src_batch = torch.full((len(neg_batch),), pos_src[idx])
                    dst_batch = torch.tensor(neg_batch)
                    new_edges = torch.stack([src_batch, dst_batch], dim=0)
                    id = torch.full((len(neg_batch),), idx)
                    new_id_list.append(id)
                    new_edges_list.append(new_edges)
                    new_labels_list.append(torch.zeros(len(neg_batch), dtype=torch.float))

                all_id = torch.cat([all_id, *new_id_list])
                all_edge_index = torch.cat([all_edge_index, *new_edges_list], dim=1)
                all_labels = torch.cat([all_labels, *new_labels_list])
            else:
                raise ValueError('Invalid split mode.')

            self.max_node_till_now = max(self.max_node_till_now, edge_index.max().item() + 1,
                                         all_edge_index.max().item() + 1)
            all_x = torch.stack([self.random_node_pe[n] for n in range(self.max_node_till_now)])

            data = Data(x=all_x, edge_index=edge_index, pos_and_neg_edge_index=all_edge_index, y=all_labels,
                        start_time=t[-1], msg=msg, id=all_id)
            snapshots.append(data)

        return snapshots
