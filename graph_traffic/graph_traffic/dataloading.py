import os
import ssl
from six.moves import urllib
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset, DataLoader
from graph_traffic.config import data_path

graph_data_path = os.path.join(data_path, "05-graph-data")


def download_file(dataset, directory):
    print("Start Downloading data: {}".format(dataset))
    url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/{}".format(
        dataset)
    print("Start Downloading File....")
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)
    with open(f"{graph_data_path}/{directory}/{dataset}", "wb") as handle:
        handle.write(data.read())


class SnapShotDataset(Dataset):
    def __init__(self, dataset_dir, npz_file, n_data_points=None):
        path = graph_data_path + "/" + dataset_dir
        if not os.path.exists(path + '/' + npz_file):
            if not os.path.exists(path):
                os.mkdir(path)
            download_file(npz_file, dataset_dir)
        zipfile = np.load(path + '/' + npz_file)
        self.x = zipfile['x']
        self.y = zipfile['y']
        if n_data_points is not None:
            self.x = self.x[:n_data_points]
            self.y = self.y[:n_data_points]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx, ...], self.y[idx, ...]


graph_dict = {
    "metr_la": "graph_la.bin",
    "pems_bay": "graph_bay.bin",
    "madrid": "madrid-graph.bin"
}


def graph_dataset(dataset):
    graph_name = graph_dict.get(dataset, "graph.bin")
    if not os.path.exists(f'{graph_data_path}/{dataset}-dataset/{graph_name}'):
        if not os.path.exists(f'{graph_data_path}/{dataset}-dataset'):
            os.mkdir(f'{graph_data_path}/{dataset}-dataset')
        download_file(graph_name, f'{dataset}-dataset')
    g, _ = dgl.load_graphs(f'{graph_data_path}/{dataset}-dataset/{graph_name}')
    return g[0]


class npzDataset(SnapShotDataset):
    def __init__(self, dataset, part, n_data_points=None):
        super(npzDataset, self).__init__(f'{dataset}-dataset', f'{dataset}_{part}.npz', n_data_points)
        if part == "train":
            self.min = np.nanmin(self.x, axis=(0, 1, 2))
            self.max = np.nanmax(self.x, axis=(0, 1, 2))


