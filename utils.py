import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
from torch_geometric.data import Data
from rdkit import Chem
import os
import numpy as np

class EnhancedGNNModel(nn.Module):
    def __init__(self, num_atom_features, num_bond_features, hidden_dim, dropout_rate=0.5):
        super(EnhancedGNNModel, self).__init__()
        self.conv1 = GCNConv(num_atom_features, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_name, protein):
    model_path = os.path.join('models', f'{model_name}_{protein}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model_state_dict = torch.load(model_path)
    if model_name == 'GCN':
        model = GNNModel(num_features=4, hidden_dim=128)
    elif model_name == 'GCN+GAT':
        # Rename keys in state dictionary
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if 'conv2.lin_src' in key:
                new_key = key.replace('conv2.lin_src', 'conv2.lin')
                new_state_dict[new_key] = value
            elif 'conv2.lin_dst' in key:
                continue  # Skip this key as it's not needed
            else:
                new_state_dict[key] = value
        
        model = EnhancedGNNModel(num_atom_features=4, num_bond_features=5, hidden_dim=128)
        model.load_state_dict(new_state_dict)
    model.eval()
    return model


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)

    num_atoms = molecule.GetNumAtoms()
    atom_features = np.zeros((num_atoms, 4))  # Include Boron as per your training script

    for atom in molecule.GetAtoms():
        atom_type = atom.GetSymbol()
        if atom_type == 'C':
            atom_features[atom.GetIdx(), 0] = 1
        elif atom_type == 'O':
            atom_features[atom.GetIdx(), 1] = 1
        elif atom_type == 'N':
            atom_features[atom.GetIdx(), 2] = 1
        elif atom_type == 'B':  # Include Boron
            atom_features[atom.GetIdx(), 3] = 1

    bond_indices = []
    bond_features = []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_indices.extend([(start, end), (end, start)])
        bond_type = bond.GetBondType()
        is_in_ring = bond.IsInRing()
        bond_feature = [
            1 if bond_type == Chem.rdchem.BondType.SINGLE else 0,
            1 if bond_type == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bond_type == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bond_type == Chem.rdchem.BondType.AROMATIC else 0,
            1 if is_in_ring else 0
        ]
        bond_features.extend([bond_feature, bond_feature.copy()])

    edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(bond_features, dtype=torch.float32)
    
    return Data(x=torch.tensor(atom_features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)
