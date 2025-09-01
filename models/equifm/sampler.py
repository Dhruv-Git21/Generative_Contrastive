# Generation sampling for EquiFM is implemented as EquivariantFlowMatchingModel.sample() in the model class.
# gen/models/equifm/sampler.py
"""
Sampling utilities for Equivariant Flow Matching.
Provides Heun (improved Euler) and Euler samplers over a fixed horizon.
"""
from typing import List, Tuple, Literal
import torch
from rdkit import Chem
from models.cvae3d.post import atoms_to_smiles  # reuse 3D->SMILES helper


def _discretize(types_logits: torch.Tensor) -> torch.Tensor:
    """Convert per-atom logits to one-hot types by argmax."""
    t_idx = torch.argmax(types_logits, dim=-1)
    one_hot = torch.zeros_like(types_logits)
    one_hot[torch.arange(types_logits.size(0)), t_idx] = 1.0
    return one_hot


@torch.no_grad()
def sample_molecule(
    model,
    context_embed: torch.Tensor,
    num_atoms: int = 12,
    steps: int = 80,
    step_size: float = 0.08,
    method: Literal["heun", "euler"] = "heun",
):
    """
    Simulate a flow from a noisy initial state to a molecule.
    Returns a SMILES string (or 'INVALID').
    """
    device = next(model.parameters()).device
    x = torch.randn((num_atoms, 3), device=device)  # positions
    y = torch.zeros((num_atoms, model.num_atom_types), device=device)  # type logits init
    y[:, 0] = 10.0  # bias toward dummy initially

    if method == "euler":
        for _ in range(steps):
            vx, vy = model(x, _discretize(y), context_embed)
            x = x + step_size * vx
            y = y + step_size * vy
    else:  # heun
        for _ in range(steps):
            vx1, vy1 = model(x, _discretize(y), context_embed)
            x1 = x + step_size * vx1
            y1 = y + step_size * vy1
            vx2, vy2 = model(x1, _discretize(y1), context_embed)
            x = x + 0.5 * step_size * (vx1 + vx2)
            y = y + 0.5 * step_size * (vy1 + vy2)

    # map atoms to symbols and build SMILES
    y_onehot = _discretize(y).detach().cpu()
    # index 0 assumed dummy; simple symbol map:
    symbol_map = {1: 'C', 2: 'N', 3: 'O', 4: 'F'}
    atoms = []
    for i in range(num_atoms):
        idx = int(torch.argmax(y_onehot[i]).item())
        if idx == 0:
            continue
        pos = x[i].detach().cpu().numpy().tolist()
        atoms.append((symbol_map.get(idx, 'C'), pos[0], pos[1], pos[2]))
    smi = atoms_to_smiles(atoms) or "INVALID"
    return smi
