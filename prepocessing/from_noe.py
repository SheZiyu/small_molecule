"""Bring the data from noe into a form that is compatible with our model. To try on the noe data set
"""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType, Mol

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}


def rdmol_to_edge(mol: Mol, include_smiles: bool = False):
    """Given mol data create a pytorch geometric data object.
    Function template taken from (https://github.com/DeepGraphLearning/ConfGF/blob/38aeb6c7719343d13fa867f4b17b02ed45d09bd0/confgf/dataset/dataset.py#L28)
    Args:
        mol (Mol): mol object
        include_smiles: should also the smiles string be included
    Returns:
        data: pytorch geometric data object
    """
    # positions = []
    # for conf in mol.GetConformers():
    #    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    #    positions.append(pos)
    # stacked_positions = torch.stack(positions, dim=2)
    N = mol.GetNumAtoms()
    # print(N)
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
    atomic_number_tensor = torch.tensor(atomic_number, dtype=torch.long)
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    return edge_index, edge_type,atomic_number_tensor


if __name__ == "__main__":
    # read mol object
    # rdkit_mol = Chem.MolFromMol2File("/home/ziyu/repos/small_molecule/data/alanine.mol2", sanitize=False, removeHs=False)
    smiles = "CC(C(=O)O)N"  # SMILES for alanine
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    edge_index, edge_type = rdmol_to_edge(mol)

    print(edge_index, edge_type)
