import pandas as pd
import csv
from rdkit import Chem
from common import metrics
from common import chem as chem_utils

def evaluate_results(generated_csv, cond_csv, train_smiles_file=None, embed_model_path=None):
    """
    Evaluate generated molecules against validity, uniqueness, novelty, and embedding similarity to target.
    - generated_csv: path to CSV with columns ['label', 'smiles']
    - cond_csv: path to original condition data CSV (to retrieve target embeddings per label)
    - train_smiles_file: (optional) path to file containing training set SMILES for novelty calculation
    - embed_model_path: (optional) path to a pre-trained embedding model if needed for computing embeddings (not used in this placeholder).
    Outputs a CSV report with metrics per label.
    """
    gen_df = pd.read_csv(generated_csv)
    cond_df = pd.read_csv(cond_csv)
    # Prepare training set SMILES for novelty if provided
    train_smiles_set = None
    if train_smiles_file:
        with open(train_smiles_file, 'r') as f:
            train_smiles = [line.strip() for line in f]
            train_smiles_set = set(train_smiles)
    # Map label to target embedding (as list of floats) from cond_df
    label_to_embed = {}
    if 'label' in cond_df.columns:
        # If multiple entries per label, use their average embedding as target
        for lbl, group in cond_df.groupby('label'):
            if 'embedding' in cond_df.columns:
                vecs = []
                for emb in group['embedding']:
                    # parse embedding similarly to dataset parsing
                    if isinstance(emb, str):
                        emb_str = emb.strip('[]() ')
                        parts = emb_str.replace(',', ' ').split()
                        vec = [float(x) for x in parts] if parts else []
                    elif isinstance(emb, (list, tuple)):
                        vec = [float(x) for x in emb]
                    else:
                        vec = []
                    if vec:
                        vecs.append(vec)
                if vecs:
                    import numpy as np
                    mean_vec = np.mean(np.array(vecs), axis=0)
                    label_to_embed[lbl] = mean_vec.tolist()
    # Evaluate metrics per label
    metrics_rows = []
    for lbl, group in gen_df.groupby('label'):
        smiles_list = group['smiles'].astype(str).tolist()
        # Compute metrics
        val = metrics.validity(smiles_list)
        uniq = metrics.uniqueness(smiles_list)
        nov = metrics.novelty(smiles_list, train_smiles_set)
        avg_dist = None
        if lbl in label_to_embed:
            target_emb = None
            try:
                import torch
                target_emb = torch.tensor(label_to_embed[lbl], dtype=torch.float)
            except:
                target_emb = None
            if target_emb is not None:
                dists = []
                for smi in smiles_list:
                    mol = chem_utils.smiles_to_mol(smi)
                    if mol:
                        emb = metrics.compute_contrastive_embedding(mol)
                        if emb is not None:
                            dists.append(metrics.embedding_distance(emb, target_emb))
                if dists:
                    avg_dist = sum(dists) / len(dists)
        metrics_rows.append({
            'label': lbl,
            'validity': val,
            'uniqueness': uniq,
            'novelty': nov,
            'avg_embed_distance': avg_dist if avg_dist is not None else ""
        })
    # Save metrics to CSV file
    out_file = generated_csv.replace('.csv', '_metrics.csv')
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'validity', 'uniqueness', 'novelty', 'avg_embed_distance'])
        for row in metrics_rows:
            writer.writerow([row['label'], f"{row['validity']:.3f}", f"{row['uniqueness']:.3f}",
                             f"{row['novelty']:.3f}", f"{row['avg_embed_distance']:.4f}" if row['avg_embed_distance'] != "" else ""])
    print(f"Evaluation metrics saved to {out_file}")
