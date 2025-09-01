import torch
from common import metrics

def embedding_similarity(mol, target_embed):
    """
    Compute a fitness score for the molecule given the target embedding.
    We will use negative Euclidean distance between the molecule's embedding and target as the score.
    """
    if mol is None or target_embed is None:
        return -float('inf')
    # Compute molecule embedding (using the same method as dataset embeddings if possible)
    mol_embed = metrics.compute_contrastive_embedding(mol)
    if mol_embed is None:
        return -float('inf')
    # If target_embed is a numpy array, convert to tensor
    if not torch.is_tensor(target_embed):
        target_embed = torch.tensor(target_embed, dtype=torch.float)
    # Compute negative distance as similarity (higher is better)
    dist = torch.dist(mol_embed, target_embed).item()
    return -dist
