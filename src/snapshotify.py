from collections import defaultdict

import networkx as nx
import pandas as pd
import torch
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset
from torch_geometric.data import Data
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from tqdm import tqdm


class TGBDataset:
    def __init__(self, name, time_interval, num_features):
        if 'tgbl' in name:
            self.loader = LinkPropPredDataset(name=name, root="datasets", preprocess=True)
        elif 'tgbn' in name:
            self.loader = NodePropPredDataset(name=name, root="datasets", preprocess=True)
        else:
            print('value error')
            exit()
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
            'edge_feat': self.data['edge_feat'],
            'edge_label': self.data['edge_label']
        })

        # Initialize variables
        current_time = interaction_data['timestamp'].min()
        end_time = interaction_data['timestamp'].max()
        snapshots = []
        pbar = tqdm(desc='Creating snapshots')
        random_node_pe = defaultdict(lambda: torch.randn(self.num_features))

        while current_time < end_time:
            next_time = current_time + self.time_interval
            mask = (interaction_data['timestamp'] >= current_time) & (interaction_data['timestamp'] < next_time)
            sub_df = interaction_data[mask]
            if len(sub_df) > 0:
                G = nx.from_pandas_edgelist(sub_df, 'source', 'target')
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                x = torch.stack([random_node_pe[n] for n in G.nodes])
                y = torch.ones(len(list(G.nodes)))
                data = Data(x=x, edge_index=edge_index, y=y)  # TODO: add negative sampling
                snapshots.append(data)
                # TODO: change temporal encoding based on the graph snapshots spacing

            current_time = next_time
            pbar.update()

        return snapshots

    def get_dataset(self, train_ratio=0.9):
        train_part = int(train_ratio * len(self.snapshot))
        return self.snapshot[:train_part], self.snapshot[train_part:]
