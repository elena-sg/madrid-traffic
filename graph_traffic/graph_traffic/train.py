import numpy as np
import torch
import torch.nn as nn
import dgl
from graph_traffic.utils import get_learning_rate

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
