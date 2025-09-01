# Conversion from density grid to atoms is implemented as density_to_atoms in generate/runner.py.
# If more sophisticated peak picking or bonding inference were required, they could be implemented here.
# gen/models/cvae3d/post.py
"""
3D post-processing utilities:
- voxel grid peak picking
- simple bonding by distance
- convert to RDKit Mol / SMILES
You can import these in generate/runner.py to replace the lightweight versions.
"""
from typing import List, Tuple
import numpy as np
from rdkit import Chem


BASIC_ATOMS = ['C', 'N', 'O', 'F', 'S']


def nms_3d(channel: np.ndarray, threshold: float = 0.5, radius: int = 1) -> List[Tuple[int, int, int]]:
    """3D non-maximum suppression on a single channel."""
    gs = channel.shape[0]
    peaks = []
    for i in range(gs):
        for j in range(gs):
            for k in range(gs):
                v = channel[i, j, k]
                if v < threshold:
                    continue
                i0, i1 = max(0, i - radius), min(gs, i + radius + 1)
                j0, j1 = max(0, j - radius), min(gs, j + radius + 1)
                k0, k1 = max(0, k - radius), min(gs, k + radius + 1)
                patch = channel[i0:i1, j0:j1, k0:k1]
                if v >= patch.max():
                    peaks.append((i, j, k))
    return peaks


def density_to_atoms(density_grid: np.ndarray, threshold: float = 0.5, nms_radius: int = 1):
    """
    Convert multi-channel voxel density into atom candidates.
    Returns a list of (symbol, x, y, z) in grid coordinates centered around 0.
    """
    num_types, gs, _, _ = density_grid.shape
    origin = (gs - 1) / 2.0
    atoms = []
    for t in range(min(num_types, len(BASIC_ATOMS))):
        peaks = nms_3d(density_grid[t], threshold=threshold, radius=nms_radius)
        for (i, j, k) in peaks:
            x, y, z = i - origin, j - origin, k - origin
            atoms.append((BASIC_ATOMS[t], float(x), float(y), float(z)))
    return atoms


def atoms_to_rdkit(atoms: List[Tuple[str, float, float, float]], bond_threshold: float = 1.65):
    """Create an RDKit Mol from atoms and simple distance-based bonding."""
    if not atoms:
        return None
    mol = Chem.RWMol()
    for sym, *_ in atoms:
        mol.AddAtom(Chem.Atom(sym))
    # set a conformer so we can measure distances
    conf = Chem.Conformer(len(atoms))
    for idx, (_, x, y, z) in enumerate(atoms):
        conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)
    # naive bonding
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
            if d < bond_threshold:
                try:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                except Exception:
                    pass
    final = mol.GetMol()
    try:
        Chem.SanitizeMol(final)
    except Chem.SanitizeException:
        pass
    return final


def atoms_to_smiles(atoms: List[Tuple[str, float, float, float]], bond_threshold: float = 1.65):
    mol = atoms_to_rdkit(atoms, bond_threshold=bond_threshold)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None
