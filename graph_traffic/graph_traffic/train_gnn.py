from functools import partial

import numpy as np
import torch
import torch.nn as nn
import dgl
from matplotlib import pyplot as plt

from graph_traffic.dataloading import graph_dataset, npzDataset
from graph_traffic.dcrnn import DiffConv
from graph_traffic.get_data import get_data, plot_graph
from graph_traffic.model import GraphRNN
from graph_traffic.utils import get_learning_rate, NormalizationLayer, masked_mae_loss
from graph_traffic.config import project_path
from datetime import datetime
import pickle
import os
from torch.utils.data import DataLoader
batch_cnt = [0]


def padding(x, y, batch_size):
    # Padding: Since the diffusion graph is precomputed we need to pad the batch so that
    # each batch have same batch size
    if x.shape[0] != batch_size:
        x_buff = torch.zeros(
            batch_size, x.shape[1], x.shape[2], x.shape[3])
        y_buff = torch.zeros(
            batch_size, x.shape[1], x.shape[2], x.shape[3])
        x_buff[:x.shape[0], :, :, :] = x
        x_buff[x.shape[0]:, :, :, :] = x[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
        y_buff[:x.shape[0], :, :, :] = y
        y_buff[x.shape[0]:, :, :, :] = y[-1].repeat(batch_size - x.shape[0], 1, 1, 1)
        x = x_buff
        y = y_buff
    return x, y


def prepare_data(x, y, batch_size, graph, normalizer, device):
    x, y = padding(x, y, batch_size)
    # Permute the dimension for shaping
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    x_norm = normalizer.normalize(x).reshape(
        x.shape[0], -1, x.shape[3]).float().to(device)
    y_norm = normalizer.normalize(y).reshape(
        x.shape[0], -1, x.shape[3]).float().to(device)
    y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)
    batch_graph = dgl.batch([graph] * batch_size)
    y = y[:, :, [0]]
    return x, y, x_norm, y_norm, batch_graph


def predict(x, y, batch_size, graph, model, device, normalizer):
    x, y, x_norm, y_norm, batch_graph = prepare_data(x, y, batch_size, graph, normalizer, device)
    output = model(batch_graph, x_norm, y_norm, batch_cnt[0], device)
    output = output[:, :, [0]]
    # Denormalization for loss compute
    y_pred = normalizer.denormalize(output)
    return y, y_pred


def train(model, graph, dataloader, optimizer, scheduler, normalizer, loss_fn, device, batch_size, max_grad_norm, minimum_lr):
    total_loss = []
    graph = graph.to(device)
    model.train()
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        y, y_pred = predict(x, y, batch_size, graph, model, device, normalizer)
        loss = loss_fn(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > minimum_lr:
            scheduler.step()
        total_loss.append(float(loss))
        batch_cnt[0] += 1
        print("Batch: ", i, end="\r")
    return np.mean(total_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, batch_size):
    total_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = batch_size
    for i, (x, y) in enumerate(dataloader):
        y, y_pred = predict(x, y, batch_size, graph, model, device, normalizer)
        loss = loss_fn(y_pred, y)
        total_loss.append(float(loss))
    return np.mean(total_loss)


def get_data_loaders(dataset_name, n_points, batch_size, num_workers):
    g = graph_dataset(dataset_name)
    train_data = npzDataset(dataset_name, "train", n_points)
    test_data = npzDataset(dataset_name, "test", n_points)

    print("Shape of train_x:", train_data.x.shape)
    print("Shape of train_y:", train_data.y.shape)
    print("Shape of test_x:", test_data.x.shape)
    print("Shape of test_y:", test_data.y.shape)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return g, train_data, test_data, train_loader, test_loader


def train_with_args(args, data_dict, meteo_dict, temporal_dict):
    training_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dict["dataset_name"] = training_time

    n_points = args["n_points"]
    batch_size = args["batch_size"]
    diffsteps = args["diffsteps"]
    decay_steps = args["decay_steps"]
    lr = args["lr"]
    minimum_lr = args["minimum_lr"]
    epochs = args["epochs"]
    max_grad_norm = args["max_grad_norm"]
    num_workers = args["num_workers"]
    model = args["model"]
    gpu = args["gpu"]
    out_feats = args["out_feats"]
    num_layers = args["num_layers"]

    if gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu))

    ids_list = data_dict["ids_list"]
    dataset_name = data_dict["dataset_name"]

    training_folder = f"{project_path}/training_history/{training_time}"
    if not os.path.exists(training_folder):
        os.mkdir(training_folder)

    with open(f"{training_folder}/learning_args.pkl", "wb") as f:
        pickle.dump(args, f)
    with open(f"{training_folder}/data_dict.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    with open(f"{training_folder}/meteo_dict.pkl", "wb") as f:
        pickle.dump(meteo_dict, f)
    with open(f"{training_folder}/temporal_dict.pkl", "wb") as f:
        pickle.dump(temporal_dict, f)

    _, _, g = get_data(data_dict, meteo_dict, temporal_dict)
    plot_graph(g, ids_list, save_dir=training_folder)

    g, train_data, test_data, train_loader, test_loader = get_data_loaders(dataset_name, n_points, batch_size, num_workers)

    seq_len = train_data.x.shape[1]
    in_feats = train_data.x.shape[-1]

    normalizer = NormalizationLayer(train_data.min, train_data.max)

    if model == "dcrnn":
        batch_g = dgl.batch([g] * batch_size).to(device)
        out_gs, in_gs = DiffConv.attach_graph(batch_g, diffsteps)
        net = partial(DiffConv, k=diffsteps, in_graph_list=in_gs, out_graph_list=out_gs)
    elif model == 'gaan':
        print("not available")

    dcrnn = GraphRNN(in_feats=in_feats,
                     out_feats=out_feats,
                     seq_len=seq_len,
                     num_layers=num_layers,
                     net=net,
                     decay_steps=decay_steps).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss

    train_losses = []
    test_losses = []
    for e in range(epochs):
        train(dcrnn, g, train_loader, optimizer, scheduler, normalizer, loss_fn, device, batch_size, max_grad_norm,
              minimum_lr)
        train_loss = eval(dcrnn, g, train_loader, normalizer, loss_fn, device, batch_size)
        test_loss = eval(dcrnn, g, test_loader, normalizer, loss_fn, device, batch_size)
        print(f"Epoch: {e} Train Loss: {train_loss} Test Loss: {test_loss}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(train_losses, label="train")
        ax.plot(test_losses, label="test")
        plt.legend()
        plt.savefig(f"{training_folder}/learning_curve.svg")
        plt.close(fig)

    print("Training finished")

    os.mkdir(f"{training_folder}/losses")
    with open(f"{training_folder}/losses/train.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open(f"{training_folder}/losses/test.pkl", "wb") as f:
        pickle.dump(test_losses, f)

    torch.save(dcrnn.state_dict(), f"{training_folder}/model.pt")


def test_model(name):
    _, _, _, train_loader, test_loader = get_data_loaders(name, None, 64, 0)
    training_folder = f"{project_path}/training_history/{name}"
