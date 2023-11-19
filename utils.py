import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
import os

def load_model(model_name, protein):
    # model_path = os.path.join('models', f'{model_name}_{protein}.pth')
    # model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_path = r'models\gnn_model_2convlayer_5fold_withmorefeatures.pth'  # Adjust the path as needed
    model_state_dict = torch.load(r"models\gnn_model_2convlayer_5fold_withmorefeatures.pth")
    model = GNNModel(num_features=4, hidden_dim=128)
    # model.load_state_dict(model_state_dict)
    # model.eval()
    model.load_state_dict(torch.load(r"models\gnn_model_2convlayer_5fold_withmorefeatures.pth"))
    model.eval()
    return model


# GNN model class
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)

    num_atoms = molecule.GetNumAtoms()

    # simple feature representation: one-hot encoding for atom types (C, O, N)
    atom_features = torch.zeros((num_atoms, 3), dtype=torch.float32)

    for atom in molecule.GetAtoms():
        atom_type = atom.GetSymbol()
        if atom_type == 'C':
            atom_features[atom.GetIdx()][0] = 1
        elif atom_type == 'O':
            atom_features[atom.GetIdx()][1] = 1
        elif atom_type == 'N':
            atom_features[atom.GetIdx()][2] = 1

    bond_indices = []
    bond_features = []

    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_indices.extend([(start, end), (end, start)])  # Add edges for both directions
        bond_features.extend([1.0, 1.0])  # Simple bond feature

    edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(bond_features, dtype=torch.float32)

    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

