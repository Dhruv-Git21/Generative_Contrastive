import logging
from rdkit import Chem
from common import chem as chem_utils
from models.rga.algorithm import genetic_optimize
from models.rga.scoring import embedding_similarity

def generate(model, model_name, dataset, output_file, num_samples=10, target_label=None, device=None):
    """
    Generate molecules using the specified model and dataset of conditions.
    If target_label is provided, generate only for that label; otherwise for all unique labels in dataset.
    Saves generated molecules to output_file (CSV with columns: label, smiles).
    """
    results = []
    labels_to_generate = []
    if target_label:
        labels_to_generate = [target_label]
    else:
        # All unique labels in dataset (if labels exist)
        labels_to_generate = sorted(set(dataset.labels)) if dataset.labels else [None]
    # Prepare label conditioner to get embedding for each label (e.g., class centroid if multiple samples per label)
    from data.conditioners import LabelConditioner
    conditioner = LabelConditioner(dataset)
    for lbl in labels_to_generate:
        # Determine condition embedding vector
        if lbl is not None:
            cond_emb = conditioner.get_condition(lbl)
            if cond_emb is None:
                # If label not in dataset or no averaging done, fallback: use first occurrence's embedding
                try:
                    idx = dataset.labels.index(lbl)
                    cond_emb = dataset.embeddings[idx]
                except:
                    logging.warning(f"No data for label {lbl} – skipping generation for this label.")
                    continue
        else:
            # If no labels (unconditional dataset scenario), just use each entry's embedding sequentially
            cond_emb = None

        # If we have cond_emb as list, convert to the appropriate tensor on device
        cond_vec = None
        if cond_emb is not None:
            import torch
            cond_vec = torch.tensor(cond_emb, dtype=torch.float).unsqueeze(0)  # shape (1, embed_dim)
            if model is not None:
                cond_vec = cond_vec.to(device)
        # If cond_emb is None (no labels scenario), we'll iterate through dataset entries individually
        if lbl is None and cond_emb is None:
            # No specific label grouping; generate for each sample in dataset
            for i in range(len(dataset)):
                cond_vec = None
                if dataset.embeddings:
                    cond_vec = dataset.embeddings[i]
                    cond_vec = torch.tensor(cond_vec, dtype=torch.float).unsqueeze(0)
                    if model is not None:
                        cond_vec = cond_vec.to(device)
                smiles_list = generate_for_condition(model, model_name, cond_vec, num_samples, device)
                label_val = dataset.labels[i] if dataset.labels else f"sample_{i}"
                for smi in smiles_list:
                    results.append((label_val, smi))
        else:
            # Generate num_samples molecules for this label condition
            smiles_list = generate_for_condition(model, model_name, cond_vec, num_samples, device)
            for smi in smiles_list:
                results.append((lbl, smi))
    # Write results to CSV
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'smiles'])
        for lbl, smi in results:
            writer.writerow([lbl if lbl is not None else "", smi if smi is not None else ""])
    logging.info(f"Generated {len(results)} molecules for {len(labels_to_generate)} condition(s).")

def generate_for_condition(model, model_name, cond_vec, num_samples, device):
    """Helper to generate a list of SMILES for a single condition vector using the specified model."""
    smiles_list = []
    if model_name == 'rga':
        # Use the genetic algorithm to generate one molecule (best) per run, run it multiple times for diversity
        # We need an initial population to start GA. For simplicity, use some molecules from dataset or a random set.
        # Here, we'll just use a simple preset small molecules set as initial pop.
        from rdkit.Chem import rdchem
        initial_smiles = ["CC", "CCO", "C1CC1", "CCN", "CCC"]  # simple seed molecules
        initial_population = [Chem.MolFromSmiles(smi) for smi in initial_smiles if Chem.MolFromSmiles(smi)]
        # If condition vector provided as tensor, get numpy for similarity calc
        target_embed = None
        if cond_vec is not None:
            target_embed = cond_vec.detach().cpu().numpy().flatten()
        for i in range(num_samples):
            best_mol = genetic_optimize(initial_population, target_embed, generations=50)
            if best_mol:
                smi = chem_utils.mol_to_smiles(best_mol)
            else:
                smi = None
            smiles_list.append(smi if smi else "INVALID")
    else:
        # Use the model's sampling method
        for i in range(num_samples):
            if model_name == 'graphaf':
                mol = model.sample_graph(cond_vec)  # returns an RDKit Mol or graph
                # If sample_graph returns a RDKit Mol, convert to SMILES
                if isinstance(mol, Chem.rdchem.Mol):
                    smi = chem_utils.mol_to_smiles(mol)
                else:
                    # If it returns a graph (adjacency, nodes), build RDKit Mol
                    adj, nodes = mol
                    smi = build_smiles_from_graph(adj, nodes)
                smiles_list.append(smi if smi else "INVALID")
            elif model_name == 'graphvae':
                # Sample latent from standard normal
                import torch
                z = torch.randn((1, model.latent_dim)).to(device)
                # Use model decode to get graph logits
                model.eval()
                with torch.no_grad():
                    adj_logits, node_logits = model.decode(z, cond_vec.to(device))
                model.train()
                # Convert logits to actual graph (threshold or argmax)
                adj_pred = (adj_logits.sigmoid() > 0.5).cpu().numpy()  # threshold at 0.5 for edges
                node_prob = node_logits.softmax(dim=-1).cpu().numpy()
                node_pred = node_prob.argmax(axis=-1)
                smi = build_smiles_from_graph(adj_pred, node_pred)
                smiles_list.append(smi if smi else "INVALID")
            elif model_name == 'cvae3d':
                import torch
                z = torch.randn((1, model.latent_dim)).to(device)
                model.eval()
                with torch.no_grad():
                    density_grid = model(z, cond_vec.to(device))
                model.train()
                # Convert density grid to atoms and bonds
                atom_list = []
                if density_grid is not None:
                    dens = density_grid.cpu().numpy()[0]  # shape (num_types, grid, grid, grid)
                    atom_list = density_to_atoms(dens)
                smi = atoms_to_smiles(atom_list)
                smiles_list.append(smi if smi else "INVALID")
            elif model_name == 'equifm':
                # Equivariant flow matching sampling
                atom_list = model.sample(cond_vec)  # We will implement sample in the model class
                smi = atoms_to_smiles(atom_list)
                smiles_list.append(smi if smi else "INVALID")
            else:
                smiles_list.append("INVALID")
    return smiles_list

def build_smiles_from_graph(adj_matrix, node_types):
    """Construct an RDKit molecule from adjacency matrix and node type list."""
    try:
        mol = Chem.RWMol()
        # Add atoms
        for t in node_types:
            # If t corresponds to dummy/stop type (assuming last index), break out
            # Here we assume dummy type has index corresponding to "Stop" in mapping
            # If it's numeric type (like atomic number), adjust accordingly
            if isinstance(t, int) and t == len(node_types)-1:
                continue  # skip dummy
            # This simplistic approach assumes type indices 0-... map to some elements; 
            # In practice, we would have a mapping from type index to atomic symbol.
            symbol = None
            if isinstance(t, str):
                symbol = t
            else:
                # We use a default mapping for a few indices for demonstration
                # (In a real scenario, keep mapping from dataset or model)
                default_map = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}
                symbol = default_map.get(t, 'C')
            mol.AddAtom(Chem.Atom(symbol))
        # Add bonds
        N = len(node_types)
        for i in range(N):
            for j in range(i+1, N):
                if i < len(adj_matrix) and j < len(adj_matrix) and adj_matrix[i][j] == 1:
                    # Add single bond
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                    except Exception as e:
                        continue
        new_mol = mol.GetMol()
        Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        smiles = chem_utils.mol_to_smiles(new_mol)
        return smiles
    except Exception as e:
        return None

def density_to_atoms(density_grid, threshold=0.5):
    """
    Identify atom positions and types from an atomic density grid.
    density_grid: numpy array of shape (num_types, grid_size, grid_size, grid_size).
    Returns a list of (type_symbol, x, y, z) for each detected atom.
    """
    atom_list = []
    num_types = density_grid.shape[0]
    grid_size = density_grid.shape[1]
    # Define coordinate mapping: assume grid centered at (0,0,0) for simplicity
    origin = (grid_size - 1) / 2.0
    for t in range(num_types):
        if t >= len(BASIC_ATOMS):  # skip background or dummy channels
            continue
        channel = density_grid[t]
        # Iterate over grid cells to find peaks above threshold
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if channel[i, j, k] >= threshold:
                        # Check if local maximum in a 3x3 neighborhood
                        val = channel[i, j, k]
                        local_max = True
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                for dk in range(-1, 2):
                                    ii, jj, kk = i+di, j+dj, k+dk
                                    if 0 <= ii < grid_size and 0 <= jj < grid_size and 0 <= kk < grid_size:
                                        if channel[ii, jj, kk] > val:
                                            local_max = False
                                            break
                                if not local_max:
                                    break
                            if not local_max:
                                break
                        if not local_max:
                            continue
                        # Compute real coordinate (assuming 1 unit per grid cell)
                        x = i - origin
                        y = j - origin
                        z = k - origin
                        atom_symbol = BASIC_ATOMS[t]
                        atom_list.append((atom_symbol, x, y, z))
    return atom_list

# Define a basic list of atom symbols corresponding to channels (for demonstration)
BASIC_ATOMS = ['C', 'N', 'O', 'F', 'S']

def atoms_to_smiles(atom_list):
    """Construct a SMILES string from a list of atoms with optional coordinates using RDKit."""
    if not atom_list:
        return None
    mol = Chem.RWMol()
    idx_list = []
    for atom_symbol, x, y, z in atom_list:
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        idx_list.append(idx)
    # Simple bonding rule: connect atoms that are within 1.6 units distance
    conf = Chem.Conformer(len(atom_list))
    for i, (_, x, y, z) in enumerate(atom_list):
        conf.SetAtomPosition(i, float(x), float(y), float(z))
    mol.GetMol().AddConformer(conf)
    for i in range(len(atom_list)):
        for j in range(i+1, len(atom_list)):
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = pos_i.Distance(pos_j)
            if dist < 1.6:  # threshold for bond
                try:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                except Exception as e:
                    continue
    new_mol = mol.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Chem.SanitizeException:
        # Sanitize may fail due to valency issues; try to adjust or remove hydrogens if needed
        try:
            Chem.Kekulize(new_mol, clearAromaticFlags=True)
        except:
            pass
    smiles = chem_utils.mol_to_smiles(new_mol)
    return smiles
