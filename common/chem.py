from rdkit import Chem

def smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit Mol object. Returns None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        mol = None
    return mol

def mol_to_smiles(mol):
    """Convert an RDKit Mol to a canonical SMILES string."""
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        smiles = None
    return smiles

def get_adjacency_and_features(mol, atom_type_to_idx):
    """
    Given an RDKit Mol, return the adjacency matrix and one-hot node feature matrix.
    atom_type_to_idx: dict mapping atom symbol to index (excluding dummy which is handled separately).
    """
    n = mol.GetNumAtoms()
    # Initialize adjacency matrix (n x n)
    adj = [[0] * n for _ in range(n)]
    # Fill adjacency for each bond (undirected graph)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # We treat any bond presence as 1 (ignoring bond order for simplicity)
        adj[i][j] = 1
        adj[j][i] = 1
    # Node features: one-hot encoding for each atom type
    features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # If atom symbol not in mapping (should not happen if mapping is complete), treat as unknown
        idx = atom_type_to_idx.get(symbol, None)
        if idx is None:
            # This can occur if a new atom type wasn't in training mapping
            idx = atom_type_to_idx.get('Unknown', None)
            # If 'Unknown' not in mapping, add it (though model won't have been trained for it)
        # Create one-hot vector for this atom
        one_hot = [0] * len(atom_type_to_idx)
        if idx is not None:
            one_hot[idx] = 1
        features.append(one_hot)
    return adj, features

def embed_molecule_3d(mol):
    """
    Generate a 3D conformer for the molecule and return a list of atomic coordinates.
    Returns None if embedding fails.
    """
    try:
        # Add hydrogens for better geometry
        mol_with_h = Chem.AddHs(mol)
        # Use ETKDG for conformation generation
        from rdkit.Chem import AllChem
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol_with_h, params)
        if result != 0:
            # Try one more time if failed
            result = AllChem.EmbedMolecule(mol_with_h, params)
        if result != 0:
            return None  # embedding failed
        # Optimize geometry
        AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=200)
        # Get coordinates of heavy atoms (excluding hydrogens for our usage, or we can keep all)
        conf = mol_with_h.GetConformer()
        coords = []
        for atom in mol_with_h.GetAtoms():
            if not atom.GetSymbol() == 'H':
                pos = conf.GetAtomPosition(atom.GetIdx())
                coords.append([pos.x, pos.y, pos.z])
        return coords
    except Exception as e:
        return None
