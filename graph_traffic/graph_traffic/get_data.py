import os
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


def get_graph(ubs, ubs_dict, weight_threshold=0.1):
    precomputed_distances = pd.read_csv(data_path + "/05-graph-data/precomputed_distances.csv", sep=";", index_col=0)
    precomputed_distances.index = precomputed_distances.index.astype(str)
    distances = pd.DataFrame(index=ubs.index.astype(str).rename("origin"),
                             columns=ubs.index.astype(str).rename("target"))
    for origin in ubs.index.astype(str).values:
        olong = ubs.loc[int(origin), "longitud"]
        olat = ubs.loc[int(origin), "latitud"]
        d = []
        for target in ubs.index.astype(str).values:
            if origin == target:
                continue
            if (origin in precomputed_distances.index.values) and\
                    (target in precomputed_distances.columns) and\
                    (not np.isnan(precomputed_distances.loc[origin, target])):
                distances.loc[origin, target] = precomputed_distances.loc[origin, target]
            else:
                tlong = ubs.loc[int(target), "longitud"]
                tlat = ubs.loc[int(target), "latitud"]
                request_path = f"http://router.project-osrm.org/route/v1/car/{olong},{olat};{tlong},{tlat}?overview=false"
                r = requests.get(request_path)  # then you load the response using the json libray
                # by default you get only one alternative so you access 0-th element of the `routes`
                routes = json.loads(r.content)
                route_1 = routes.get("routes")[0]
                distances.loc[origin, target] = route_1["duration"]
                precomputed_distances.loc[origin, str(target)] = route_1["duration"]
    precomputed_distances.to_csv(data_path + "/05-graph-data/precomputed_distances.csv", sep=";")
    distances = distances.astype(float)
    std = np.nanstd(distances.to_numpy().ravel())

    def get_weight(distance):
        return exp(-distance ** 2 / std ** 2)

    weights = distances.applymap(get_weight).fillna(0).round(4)

    weights_lim = weights[weights > weight_threshold].stack()
    nodes_src, nodes_target = zip(*weights_lim.index)
    nodes_src = np.array(nodes_src)
    nodes_target = np.array(nodes_target)

    nodes_src_graph = np.array([ubs_dict[int(x)] for x in nodes_src])
    nodes_target_graph = np.array([ubs_dict[int(x)] for x in nodes_target])

    graph_dataset = MadridTrafficDataset(ubs.shape[0], nodes_src_graph, nodes_target_graph, weights_lim)

    return graph_dataset


def get_mmagns(meteo_dict):
    mmagns = []
    if meteo_dict["rain"] != "drop":
        mmagns.append("precipitacion")
    if meteo_dict["wind"] != "drop":
        mmagns += ["dir_viento", "velocidad_viento"]
    if meteo_dict["temperature"] != "drop":
        mmagns.append("temperatura")
    if meteo_dict["humidity"] != "drop":
        mmagns.append("humedad_relativa")
    if meteo_dict["pressure"] != "drop":
        mmagns.append("presion_barometrica")
    if meteo_dict["radiation"] != "drop":
        mmagns.append("radiacion_solar")
    return mmagns


# def get_data(ids_list, seq_len, rain, wind, temperature, humidity, pressure, radiation, season, month, day_of_month,
#              hour, interactions, with_graph, from_date, to_date, dataset_name, target):
def get_data(data_dict, meteo_dict, temporal_dict, train_until=None):
    ids_list = data_dict["ids_list"]
    from_date = data_dict["from_date"]
    to_date = data_dict["to_date"]
    target = data_dict["target"]
    seq_len = data_dict["seq_len"]
    dataset_name = data_dict["dataset_name"]
    with_graph = data_dict["with_graph"]
    ubs, ubs_dict = ubs_index(ids_list)

    mmagns = get_mmagns(meteo_dict)

    dates = pd.date_range(from_date, to_date, freq="15min")
    dfs_dict = {}
    for id in ids_list:
        dfs_dict[id] = merge_data(id, from_date, to_date, target, mmagns, seq_len)
        dates = dates.intersection(dfs_dict[id].date)

    for id in ids_list:
        df = dfs_dict[id]
        df = df[df.date.isin(dates)]
        dfs_dict[id] = transform_df(df, meteo_dict, temporal_dict, data_dict["interactions"], target)

    #n_rows = dfs_dict[id].shape[0]
    n_features = dfs_dict[id].shape[1]

    right_time_gaps = (dates.to_series().diff().apply(lambda x: x.total_seconds() / 60) == 15).rolling(2*seq_len).sum() == 2*seq_len
    right_time_gaps = right_time_gaps.shift(-2 * seq_len).fillna(False).reset_index(drop=True)
    right_time_gaps = right_time_gaps[right_time_gaps].index.values

    n_rows = len(right_time_gaps)

    arrx = np.full((n_rows, seq_len, len(ids_list), n_features), np.nan)
    arry = np.full((n_rows, seq_len, len(ids_list), n_features), np.nan)
    for sensor, df in dfs_dict.items():
        graph_id = ubs_dict[sensor]
        dfi = pd.DataFrame(df)
        for i, timestamp in enumerate(right_time_gaps):
            arrx[i, :, graph_id, :] = dfi.iloc[timestamp:timestamp+seq_len]
            arry[i, :, graph_id, :] = dfi.iloc[timestamp+seq_len:timestamp + 2*seq_len]
        # for period in range(seq_len):
        #     arrx[:, period, graph_id, :] = dfi.iloc[right_time_gaps].shift(-period).values
        #     arry[:, period, graph_id, :] = dfi.iloc[right_time_gaps].shift(-seq_len-period).values

    #arrx = arrx[:-seq_len]
    #arry = arrx[seq_len:]
    #arrx = arrx[:-seq_len]

    data_size = arrx.shape[0]
    if train_until is None:
        train_data_size = int(0.8 * data_size)
    else:
        dates_train = (dates.to_series().reset_index(drop=True) <= train_until)
        train_index = np.intersect1d(dates_train[dates_train].index.values, right_time_gaps)
        train_data_size = len(train_index)

    if not os.path.exists(f"{data_path}/05-graph-data/{dataset_name}-dataset"):
        os.mkdir(f"{data_path}/05-graph-data/{dataset_name}-dataset")
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_dataset.npz", x=arrx, y=arry)
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_train.npz",
             x=arrx[:train_data_size], y=arry[:train_data_size])
    # np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_valid.npz",
    #          x=arrx[int(0.6 * data_size):int(0.8 * data_size)], y=arry[int(0.6 * data_size):int(0.8 * data_size)])
    np.savez(f"{data_path}/05-graph-data/{dataset_name}-dataset/{dataset_name}_test.npz",
             x=arrx[train_data_size:], y=arry[train_data_size:])

    if not with_graph:
        return arrx, arry
    else:
        if "graph_weight_threshold" not in data_dict.keys():
            data_dict["graph_weight_threshold"] = 0.1
        graph = get_graph(ubs, ubs_dict, weight_threshold=data_dict["graph_weight_threshold"])[0]
        dgl.save_graphs(f"{data_path}/05-graph-data/{dataset_name}-dataset/graph.bin", [graph])
        return arrx, arry, graph


def plot_graph(graph, ids_list, save_dir=None, graph_name="graph", layout=nx.spring_layout):
    _, ubs_dict = ubs_index(ids_list)
    labels_dict = {v: k for (k, v) in ubs_dict.items()}
    nx_G = graph.to_networkx()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = layout(nx_G)
    widths = 5.8 * graph.edata["weight"].numpy() - 2.8
    nx.draw(nx_G, pos, with_labels=True, labels=labels_dict, width=widths, node_size=1500)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{graph_name}.svg")
        plt.savefig(f"{save_dir}/{graph_name}.png")
        plt.clf()