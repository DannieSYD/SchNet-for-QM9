import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch import Tensor
from torch_geometric.nn import GCNConv, global_add_pool, radius_graph, MessagePassing
import pickle
import os
import random
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("PyTorch version:", torch.__version__)


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed(0)

# 1. Data Loader for QM9 dataset
def load_data():
    dataset = QM9(root='./qm9')

    target = 7
    KCALMOL2EV = 0.04336414

    atomref = dataset.atomref(target)

    index = 0
    for data in dataset:
        y_hat = data.y[:, target] - atomref[data.z].sum()
        dataset[index].y[:, target] = y_hat
        index += 1

    dataset.y = dataset.y//KCALMOL2EV
    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = (dataset.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()
    print(f"mean and std values of U0: mean = {mean}, std = {std}")

    train_dataset = dataset[:50000]
    test_dataset = dataset[50000:51000]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
    return train_loader, test_loader, mean, std


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:  # dist = d_{ij} range: 0-7
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CFConv(MessagePassing):
    def __init__(self, nn_layers: nn.Sequential, cutoff: float):
        super().__init__(aggr='add')
        self.cutoff = cutoff
        self.nn = nn_layers

    def forward(self, h, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.propagate(edge_index, x=h, W=W)  # message -> aggregate -> update
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class Interaction(nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int, cutoff: float, num_filters: int):
        super().__init__()
        self.cutoff = cutoff
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, num_filters),
            ShiftedSoftplus(),
        )
        self.atom_wise = nn.Linear(hidden_channels, hidden_channels)
        self.conv = CFConv(self.mlp, self.cutoff)
        self.out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.atom_wise.weight)
        self.atom_wise.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.out[0].weight)
        self.out[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.out[2].weight)
        self.out[2].bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, edge_attr):
        h = self.atom_wise(h)
        h = self.conv(h, edge_index, edge_weight, edge_attr)
        h = self.out(h)
        return h


class SchNetModel(nn.Module):
    def __init__(self):
        super(SchNetModel, self).__init__()
        # TODO: Define layers and modules here, for instance:
        # self.conv = SomeGeometricLayer()
        self.hidden_channels = 128
        self.num_interactions = 6
        self.num_filters = 128
        self.num_gaussians = 50
        self.cutoff = 10.0
        self.max_num_neighbors = 32

        self.embedding = nn.Embedding(100, self.hidden_channels, padding_idx=0)
        self.interactions = nn.ModuleList()
        for _ in range(self.num_interactions):
            block = Interaction(self.hidden_channels, self.num_gaussians, self.cutoff, self.num_filters)
            self.interactions.append(block)
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_channels // 2, 1)
        )
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        self.layers[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        self.layers[2].bias.data.fill_(0)

    def forward(self, data):
        pos, z, batch = data.pos, data.z, data.batch
        pos.require_grad = True

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        x_emb = self.embedding(z)
        h = x_emb
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        h = self.layers(h)
        out = global_add_pool(h, batch)
        return out


def evaluate_model(model, epoch, loader, device, mean, std):
    model.eval()
    total_loss = 0
    total_mae = 0
    for data in loader:
        data.to(device)
        with torch.no_grad():
            out = model(data)
            target = data.y[:, 7].unsqueeze(1)
            loss = F.mse_loss(out, target)

            out = out*std+mean
            target = target*std+mean
            mae = torch.sum(torch.abs(target-out)) // len(target)

            total_loss += loss.item()
            total_mae += mae.item()
    loss = total_loss / len(loader)
    mae = total_mae / len(loader)
    print(f"------------------------------------ Validation {epoch} Loss = {loss:.6f}, MAE = {mae:.6f}")
    return loss, mae


def verify_permutation_invariance(model, data):
    permuted_data = data.clone()  # Clone to get a new copy
    node_permutation = torch.randperm(data.num_nodes)
    permuted_data.x = data.x[node_permutation]
    if data.edge_index is not None:
        edge_index_remap = {i: node_permutation[i].item() for i in range(data.num_nodes)}
        permuted_data.edge_index = torch.tensor([[edge_index_remap[i.item()] for i in row] for row in data.edge_index.t()]).t()

    return torch.allclose(model(data), model(permuted_data))


def rotate_molecule(coordinates, angle, axis):
    R = torch.eye(3)
    angle = torch.tensor(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    if axis == 0:  # Rotate around x-axis
        R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    elif axis == 1:  # Rotate around y-axis
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    elif axis == 2:  # Rotate around z-axis
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return torch.matmul(coordinates, R)


def verify_rotation_invariance(model, data):
    rotated_data = data.clone() # Clone to get a new copy
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.choice([0, 1, 2])
    rotated_data.pos = rotate_molecule(data.pos, angle, axis)

    return torch.allclose(model(data), model(rotated_data))


device = torch.device('cuda')
model = SchNetModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)

pretrained = False
model_path = 'schnet_energy_model_100.pth'
dataset = QM9(root='./qm9')
train_loader, test_loader, mean, std = load_data()

data = train_loader.dataset[0]
print("Let us print all the attributes (along with their shapes) that our PyG molecular graph contains:")
print(data)

global_step = 0
mse = torch.zeros(10000)
mse_test = torch.zeros(10000)
mae = torch.zeros(10000)
mae_test = torch.zeros(10000)

if not pretrained:
    for epoch in range(10000):
        total_loss = 0
        total_mae = 0
        model.train()
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = model(data)
            target = data.y[:, 7].unsqueeze(1)

            loss = F.mse_loss(out, target)  # Using the 7th property as an example target
            loss.backward()
            optimizer.step()

            out = out * std + mean
            target = target * std + mean
            mae_e = torch.sum(torch.abs(target - out)) // len(target)
            total_loss += loss.item()
            total_mae += mae_e.item()
            global_step += 1

            if global_step % 10000 == 0:
                scheduler.step()

        mse[epoch] = total_loss / len(train_loader)
        mae[epoch] = total_mae / len(train_loader)
        print(f"Training {epoch} Loss = {loss:.6f}, MAE = {mae_e:.6f}, LR = {scheduler.get_last_lr()}")

        # Evaluate
        mse_test[epoch], mae_test[epoch] = evaluate_model(model, epoch, test_loader, device, mean, std)
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1} Model Saving ...")
            torch.save(model.state_dict(), f"schnet_energy_model_{epoch+1}.pth")
else:
    model = SchNetModel()
    model.load_state_dict(torch.load(model_path))

sample_data = next(iter(test_loader))
print("Permutation Invariance:", verify_permutation_invariance(model, sample_data))
print("Rotation Invariance:", verify_rotation_invariance(model, sample_data))

with open('mse.pkl', 'wb') as file:
    pickle.dump(mse, file)

with open('mae.pkl', 'wb') as file:
    pickle.dump(mae, file)

with open('mse_test.pkl', 'wb') as file:
    pickle.dump(mse_test, file)

with open('mae_test.pkl', 'wb') as file:
    pickle.dump(mae_test, file)

k=0
