import sys

import matplotlib.pyplot as plt

sys.path.append("C:\\Users\\Bened\\Documents\\TFM\\dgl\\examples\\pytorch\\dtgrnn")

from functools import partial
import argparse
import torch
from torch.utils.data import DataLoader
import dgl
from graph_traffic.model import GraphRNN
from graph_traffic.dcrnn import DiffConv
from graph_traffic.gaan import GatedGAT
from graph_traffic.dataloading import METR_LAGraphDataset, METR_LATrainDataset,\
    METR_LATestDataset, METR_LAValidDataset,\
    PEMS_BAYGraphDataset, PEMS_BAYTrainDataset,\
    PEMS_BAYValidDataset, PEMS_BAYTestDataset, MadridGraphDataset, \
    MadridTrainDataset, MadridTestDataset, MadridValidDataset
from graph_traffic.utils import NormalizationLayer, masked_mae_loss
from graph_traffic.train_gnn import predict

batch_cnt = [0]


parser = argparse.ArgumentParser()
# Define the arguments
parser.add_argument('--batch_size', type=int, default=64,
                    help="Size of batch for minibatch Training")
parser.add_argument('--num_workers', type=int, default=0,
                    help="Number of workers for parallel dataloading")
parser.add_argument('--model', type=str, default='dcrnn',
                    help="WHich model to use DCRNN vs GaAN")
parser.add_argument('--gpu', type=int, default=-1,
                    help="GPU indexm -1 for CPU training")
parser.add_argument('--diffsteps', type=int, default=2,
                    help="Step of constructing the diffusiob matrix")
parser.add_argument('--num_heads', type=int, default=2,
                    help="Number of multiattention head")
parser.add_argument('--decay_steps', type=int, default=2000,
                    help="Teacher forcing probability decay ratio")
parser.add_argument('--lr', type=float, default=0.01,
                    help="Initial learning rate")
parser.add_argument('--minimum_lr', type=float, default=2e-6,
                    help="Lower bound of learning rate")
parser.add_argument('--dataset', type=str, default='LA',
                    help="dataset LA for METR_LA; BAY for PEMS_BAY")
parser.add_argument('--epochs', type=int, default=100,
                    help="Number of epoches for training")
parser.add_argument('--max_grad_norm', type=float, default=5.0,
                    help="Maximum gradient norm for update parameters")
args = parser.parse_args()
# Load the datasets
if args.dataset == 'LA':
    g = METR_LAGraphDataset()
    train_data = METR_LATrainDataset()
    test_data = METR_LATestDataset()
    valid_data = METR_LAValidDataset()
    seq_len = 12
    out_feats = 64
elif args.dataset == 'BAY':
    g = PEMS_BAYGraphDataset()
    train_data = PEMS_BAYTrainDataset()
    test_data = PEMS_BAYTestDataset()
    valid_data = PEMS_BAYValidDataset()
    seq_len = 12
    out_feats = 64
elif args.dataset == "madrid":
    g = MadridGraphDataset()
    train_data = MadridTrainDataset()
    test_data = MadridTestDataset()
    valid_data = MadridValidDataset()
    seq_len = 12
    out_feats = 64

if args.gpu == -1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(args.gpu))

train_loader = DataLoader(
    train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
valid_loader = DataLoader(
    valid_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
test_loader = DataLoader(
    test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
normalizer = NormalizationLayer(train_data.mean, train_data.std)

if args.model == 'dcrnn':
    batch_g = dgl.batch([g]*args.batch_size).to(device)
    out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
    net = partial(DiffConv, k=args.diffsteps,
                  in_graph_list=in_gs, out_graph_list=out_gs)
elif args.model == 'gaan':
    net = partial(GatedGAT, map_feats=64, num_heads=args.num_heads)

dcrnn = GraphRNN(in_feats=2,
                 out_feats=out_feats,
                 seq_len=seq_len,
                 num_layers=2,
                 net=net,
                 decay_steps=args.decay_steps).to(device)
dcrnn.load_state_dict(torch.load("trained-dcrnn.pt"))

loss_fn = masked_mae_loss

for i, (x, y) in enumerate(train_loader):
    #x, y, x_norm, y_norm, batch_graph = prepare_data(g.to(device), x, y, normalizer, args.batch_size, device)
    #y_pred = predict(dcrnn, batch_graph, x_norm, y_norm, normalizer, device, i)
    dcrnn.eval()
    y, y_pred = predict(x, y, args.batch_size, g.to(device), dcrnn, device, normalizer)
    break

for i in range(10):
    fig, ax = plt.subplots()
    #ax.set_title(f"de {(y[:, i, 1]*24).min().numpy()} a  {(y[:, i, 1]*24).max().numpy()}, sensor{i%5+1}")
    ax.plot(y[:, i, 0].detach().numpy(), label="real")
    ax.plot(y_pred[:, i, 0].detach().numpy(), label="pred")
    plt.legend()
