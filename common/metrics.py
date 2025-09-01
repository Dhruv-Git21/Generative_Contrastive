import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from common import chem

def compute_contrastive_embedding(mol):
    """
    Placeholder for computing the contrastive embedding of a molecule.
    In practice, this should use the same model that generated the embeddings in the dataset.
    Here we use an RDKit 2048-bit Morgan fingerprint as a proxy (and convert to float tensor).
    """
    try:
        # Compute Morgan fingerprint bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = list(fp.ToBitString())  # list of '0'/'1'
        arr = [float(x) for x in arr]  # convert to float
        emb = torch.tensor(arr)
        return emb
    except Exception as e:
        return None

def embedding_distance(embed1, embed2):
    """Compute Euclidean distance between two embedding vectors (as PyTorch tensors)."""
    return torch.dist(embed1, embed2).item()

def validity(smiles_list):
    """Calculate fraction of valid SMILES in the list."""
    total = len(smiles_list)
    if total == 0:
        return 0.0
    valid_count = 0
    for s in smiles_list:
        if s and chem.smiles_to_mol(s):
            valid_count += 1
    return valid_count / total

def uniqueness(smiles_list):
    """Calculate fraction of unique SMILES in the list."""
    total = len(smiles_list)
    if total == 0:
        return 0.0
    unique_set = set([s for s in smiles_list if s])
    return len(unique_set) / total

def novelty(smiles_list, training_smiles_set):
    """Calculate fraction of SMILES that are not in the training set (novelty)."""
    total = len(smiles_list)
    if total == 0 or training_smiles_set is None:
        return 0.0
    novel_count = sum(1 for s in smiles_list if s and s not in training_smiles_set)
    return novel_count / total
