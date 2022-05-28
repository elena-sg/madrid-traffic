from graph_traffic.dataloading import graph_dataset, npzDataset
from graph_traffic.dcrnn import DiffConv
from graph_traffic.config import project_path
from graph_traffic.model import GraphRNN
from graph_traffic.train_gnn import train, eval
from graph_traffic.utils import NormalizationLayer, masked_mae_loss


from torch.utils.data import DataLoader
import dgl
import torch
from functools import partial

n_points = 1000
dataset_name = "001"
batch_size = 64
diffsteps = 2
decay_steps = 2000
lr = 0.01
minimum_lr = 2e-6
epochs = 5
max_grad_norm = 5.0
num_workers = 0
model = "dcrnn"
gpu = -1
num_heads = 2 # relevant for model="gaan"
out_feats = 64
num_layers = 2

if gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(gpu))

g = graph_dataset(dataset_name)
train_data = npzDataset(dataset_name, "train", n_points)
test_data = npzDataset(dataset_name, "test", n_points)
valid_data = npzDataset(dataset_name, "valid", n_points)

seq_len = train_data.x.shape[1]
in_feats = train_data.x.shape[-1]

train_loader = DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = DataLoader(
    valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = DataLoader(
    test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

normalizer = NormalizationLayer(train_data.mean, train_data.std)

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

for e in range(epochs):
    train(dcrnn, g, train_loader, optimizer, scheduler, normalizer, loss_fn, device, batch_size, max_grad_norm,
          minimum_lr)
    train_loss = eval(dcrnn, g, train_loader, normalizer, loss_fn, device, batch_size)
    valid_loss = eval(dcrnn, g, valid_loader, normalizer, loss_fn, device, batch_size)
    test_loss = eval(dcrnn, g, test_loader, normalizer, loss_fn, device, batch_size)
    print(f"Epoch: {e} Train Loss: {train_loss} Valid Loss: {valid_loss} Test Loss: {test_loss}")