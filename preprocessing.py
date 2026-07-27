# %%
import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Data


# %%
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]

# %%
def bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

# %%
def smiles_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. Extract Node Features
    x = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(x, dtype=torch.float)

    # 2. Extract Edge Features & Connections
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)

        # Both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(feat)
        edge_attr.append(feat)

    # 3. Convert to Tensors OUTSIDE the loop
    if len(edge_index) > 0:
        edge_index = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Handle zero-bond molecules (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    y = torch.tensor([target], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)




# %%
def create_dataset(df, target="Thermal_cond"):
    dataset = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row["SMILES"], row[target])
        if graph is not None:
            dataset.append(graph)

    return dataset


