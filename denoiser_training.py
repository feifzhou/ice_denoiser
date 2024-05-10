import argparse
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from update_src.transforms import RattleParticles, DownselectEdges
from sklearn.preprocessing import LabelEncoder
from update_src.transforms import PeriodicRadiusGraph
from torch_geometric.loader import DataLoader
from torch import nn
from functools import partial
from update_src.nn.basis import bessel
from update_src.nn.models.e3nn_nequip import NequIP
# from update_src.utils.utils import summary
from tqdm.notebook import trange
from denoiser_config import get_cfg_defaults
import glob
import ase.io

def atoms2pygdata(atoms, ase=False):
    le = LabelEncoder()
    x = le.fit_transform(atoms.numbers)
    if ase:
        return Data(
            x       = torch.tensor(x,               dtype=torch.long),
            pos     = torch.tensor(atoms.positions, dtype=torch.float),
            cell    = np.array(atoms.cell),
            pbc     = atoms.pbc,
            numbers = atoms.numbers,
        )
    else:
        return Data(
            x    = torch.tensor(x,                    dtype=torch.long),
            pos  = torch.tensor(atoms.positions,      dtype=torch.float),
            box  = torch.tensor(np.array(atoms.cell), dtype=torch.float).sum(dim=0),
        )

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.dataset = dataset

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class InitialEmbedding(nn.Module):
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        h_node_x = self.embed_node_x(data.x)
        h_node_z = self.embed_node_z(data.x)
        h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        data.h_node_x = h_node_x
        data.h_node_z = h_node_z
        data.h_edge   = h_edge
        return data

def loss_fn(model, data):
    disp_true = data.disp
    disp_pred = model(data)
    return torch.nn.functional.mse_loss(disp_pred, disp_true)

def train(loader, model, optimizer, device, pin_memory):
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device, non_blocking=pin_memory)
        loss = loss_fn(model, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
def plot_loss(L_train):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.plot(L_train, label='train')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    ax2.semilogy(L_train, label='train')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    plt.savefig('./training_loss_curves.png')
    return

def main():
    parser = argparse.ArgumentParser(description="Training of denoiser model")
    parser.add_argument("--cfg", default="", help="Path to the user-defined config file")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Read/construct ideal structures
    poscar_list = glob.glob(cfg.DATA.IDEAL_STRUCTURES)
    ideal_atoms_list = []
    for p in poscar_list:
        atoms = ase.io.read(p)
        atoms = atoms.repeat((3, 3, 3))
        ideal_atoms_list.append(atoms)

    # Prepare dataloader
    dataset = [atoms2pygdata(atoms, ase=False) for atoms in ideal_atoms_list]
    transform = PeriodicRadiusGraph(cutoff=cfg.CUTOFF)
    dataset = [transform(d) for d in dataset]
    dataset = [d.clone() for d in dataset for _ in range(64)]
    dataset = CustomDataset(dataset, transform=Compose([RattleParticles(sigma_max=cfg.TRAIN.SIGMA_MAX), DownselectEdges(cutoff=cfg.CUTOFF)]))
    loader_train = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=cfg.TRAIN.PIN_MEMORY)

    # Initialize the graph network model
    model = NequIP(
        init_embed=InitialEmbedding(num_species=cfg.MODEL.NUM_SPECIES, cutoff=cfg.CUTOFF),
        irreps_node_x='8x0e',
        irreps_node_z='8x0e',
        irreps_hidden=cfg.MODEL.IRREPS_HIDDEN,
        irreps_edge='1x0e + 1x1o + 1x2e',
        irreps_out='1x1o',
        num_convs=3,
        radial_neurons=[16, 64],
        num_neighbors=cfg.MODEL.NUM_NEIGH,
    )
    # summary(model).tail(1)

    # Train the model
    num_samples = len(dataset)
    num_epochs = int(cfg.TRAIN.NUM_UPDATES / (num_samples / cfg.TRAIN.BATCH_SIZE))
    print(f'{num_epochs} epochs needed to update the model {cfg.TRAIN.NUM_UPDATES} times.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARN_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L_train = []

    model = model.to(device)

    for e in trange(num_epochs, desc="Training", unit="Epochs"):
        loss_train = train(loader_train, model, optimizer, device, cfg.TRAIN.PIN_MEMORY)
        if e % 10 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train,
            }, cfg.MODEL.CKT_FILE)
        L_train.append(loss_train)

    # Save the trained model
    plot_loss(L_train)
    torch.save(model, cfg.MODEL.SAVE_FILE)
    

if __name__ == '__main__':
    main()
