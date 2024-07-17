from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset
from torch_geometric.data import Data
from tqdm import tqdm
import random


def add_negative_edges(G, edge_index):
    all_nodes = list(range(edge_index.max().item() + 1))
    num_edges = edge_index.size(1)
    negative_edges = []
    make_sure_not_positive = False
    if make_sure_not_positive:
        while len(negative_edges) < num_edges:
            u, v = random.sample(all_nodes, 2)
            if not G.has_edge(u, v):
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
            self.loader = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
        elif 'tgbn' in name:
            self.loader = NodePropPredDataset(name=name, root="datasets", preprocess=True)
        else:
            raise ValueError('Unsupported dataset name')
        self.num_features = num_features
        self.data = self.loader.full_data
        self.time_interval = time_interval
        self.snapshot = self.create_snapshots()

    def create_snapshots(self):
        # Create DataFrame
        interaction_data = pd.DataFrame({
            'source': self.data['sources'],
            'target': self.data['destinations'],
            'timestamp': self.data['timestamps'],
            'edge_label': self.data['edge_label']
        })
        edge_features = self.data['edge_feat']
        # Initialize variables
        snapshots = []
        pbar = tqdm(desc='Creating snapshots')
        random_node_pe = defaultdict(lambda: torch.randn(self.num_features))

        while len(interaction_data) > 0:
            sub_df = interaction_data.iloc[:self.time_interval]
            if len(sub_df) > 0:
                G = nx.from_pandas_edgelist(sub_df, 'source', 'target')
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                all_x = torch.stack([random_node_pe[n] for n in list(range(edge_index.max().item() + 1))])
                all_edge_index, all_labels = add_negative_edges(G, edge_index)

                data = Data(x=all_x, edge_index=all_edge_index, y=all_labels, edge_feature=edge_features,
                            start_time=sub_df.timestamp.min())
                snapshots.append(data)

            interaction_data = interaction_data.iloc[self.time_interval:]
            pbar.update()

        return snapshots

    def get_dataset(self, train_ratio=0.9):
        train_part = int(train_ratio * len(self.snapshot))
        return self.snapshot[:train_part], self.snapshot[train_part:]
