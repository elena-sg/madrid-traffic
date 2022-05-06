from datetime import datetime
import json
from math import exp
import networkx as nx
import dgl
import numpy as np
import pandas as pd
import requests
import torch
from dgl.data import DGLDataset
from matplotlib import pyplot as plt

from graph_traffic.config import data_path, project_path
from graph_traffic.merge_data import merge_data
from graph_traffic.custom_transformer import transform_df

ubs_path = f"{data_path}/01-raw/traffic/ubs.csv"

class MadridTrafficDataset(DGLDataset):
    def __init__(self, n_nodes, nodes_src_graph, nodes_target_graph, weights):
        self.edges_src = torch.from_numpy(nodes_src_graph)
        self.edges_dst = torch.from_numpy(nodes_target_graph)
        self.n_nodes = n_nodes
        self.weights = weights
        super().__init__(name='madrid_traffic')



    def process(self):
        self.graph = dgl.graph((self.edges_src, self.edges_dst), num_nodes=self.n_nodes)
        #self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = torch.from_numpy(self.weights.values)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_train = int(self.n_nodes * 0.6)
        n_val = int(self.n_nodes * 0.2)
        train_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


def ubs_index(ids_list):
    ubs = pd.read_csv(ubs_path)
    ubs = ubs[ubs["id"].isin(ids_list)].set_index("id")
    ubs_dict = ubs.reset_index()[["id"]].reset_index().set_index("id").to_dict()["index"]
    ubs["id_graph"] = ubs.index.map(lambda x: ubs_dict[x])
    return ubs, ubs_dict


def get_graph(ubs, ubs_dict):
    distances = pd.DataFrame(index=ubs.index.rename("origin"), columns=ubs.index.rename("target"))
    for origin in ubs.index.values:
        olong = ubs.loc[origin, "longitud"]
        olat = ubs.loc[origin, "latitud"]
        d = []
        for target in ubs.index.values:
            if origin == target:
                continue
            tlong = ubs.loc[target, "longitud"]
            tlat = ubs.loc[target, "latitud"]
            request_path = f"http://router.project-osrm.org/route/v1/car/{olong},{olat};{tlong},{tlat}?overview=false"
            r = requests.get(request_path)  # then you load the response using the json libray
            # by default you get only one alternative so you access 0-th element of the `routes`
            routes = json.loads(r.content)
            route_1 = routes.get("routes")[0]
            distances.loc[origin, target] = route_1["duration"]
    distances = distances.astype(float)

    # manual manipulation of distances
    if 3978 in distances.index.values:
        if 3973 in distances.index.values:
            distances.loc[3978, 3973] = 60
            distances.loc[3973, 3978] = 90
        if 3976 in distances.index.values:
            distances.loc[3976, 3978] = 150
        if 3977 in distances.index.values:
            distances.loc[3977, 3978] = 60

    std = np.nanstd(distances.to_numpy().ravel())

    def get_weight(distance):
        return exp(-distance ** 2 / std ** 2)

    weights = distances.applymap(get_weight).fillna(0).round(4)

    weights_lim = weights[weights > 0.01].stack()
    nodes_src, nodes_target = zip(*weights_lim.index)
    nodes_src = np.array(nodes_src)
    nodes_target = np.array(nodes_target)

    nodes_src_graph = np.array([ubs_dict[x] for x in nodes_src])
    nodes_target_graph = np.array([ubs_dict[x] for x in nodes_target])

    graph_dataset = MadridTrafficDataset(ubs.shape[0], nodes_src_graph, nodes_target_graph, weights_lim)

    return graph_dataset


def get_data(ids_list, seq_len, rain, wind, season, month, day_of_month, hour, interactions, with_graph, from_date,
             to_date, dataset_name):
    ubs, ubs_dict = ubs_index(ids_list)

    data_dict = {}
    for id in ids_list:
        df = merge_data(id, from_date, to_date)
        data_dict[id] = transform_df(df, rain, wind, season, month, day_of_month, hour, interactions)

    n_rows = data_dict[id].shape[0]
    n_features = data_dict[id].shape[1]

    arrx = np.full((n_rows, seq_len, len(ids_list), n_features), np.nan)
    for sensor, df in data_dict.items():
        graph_id = ubs_dict[sensor]
        dfi = pd.DataFrame(df)
        for period in range(seq_len):
            arrx[:, period, graph_id, :] = dfi.shift(-period).values

    arrx = arrx[:-seq_len]
    arry = arrx[seq_len:]
    arrx = arrx[:-seq_len]

    data_size = arrx.shape[0]

    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_dataset.npz", x=arrx, y=arry)
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_train.npz",
             x=arrx[:int(0.6 * data_size)], y=arry[:int(0.6 * data_size)])
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_valid.npz",
             x=arrx[int(0.6 * data_size):int(0.8 * data_size)], y=arry[int(0.6 * data_size):int(0.8 * data_size)])
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_test.npz",
             x=arrx[int(0.8 * data_size):], y=arry[int(0.8 * data_size):])

    if not with_graph:
        return arrx, arry
    else:
        graph = get_graph(ubs, ubs_dict)[0]
        dgl.save_graphs(f"{data_path}/05-graph-data/{dataset_name}-dataset/graph.bin", [graph])
        return arrx, arry, graph


def plot_graph(graph, ids_list, save=False):
    _, ubs_dict = ubs_index(ids_list)
    labels_dict = {v: k for (k, v) in ubs_dict.items()}
    nx_G = graph.to_networkx()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, labels=labels_dict, width=graph.edata["weight"].numpy() * 30, node_size=1500)
    if save:
        plt.savefig(f"{project_path}/plots/graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")