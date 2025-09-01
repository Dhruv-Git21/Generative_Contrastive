import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from functools import partial
import numpy as np
import torch


from common import chem as chem_utils

SMILES_CANDIDATES = ["smiles", "SMILES", "canonical_smiles", "can_smiles", "smile", "Smiles"]

def _get_smiles_list(df: pd.DataFrame, preferred: Optional[str] = None):
    """
    Try to find a SMILES column. Returns (list_of_smiles, column_name or None).
    Filters obvious NaN-like strings.
    """
    order = []
    if preferred:
        order.append(preferred)
    order.extend([c for c in SMILES_CANDIDATES if c not in order])
    for c in order:
        if c in df.columns:
            series = df[c].astype(str)
            series = series[series.str.lower() != "nan"]  # drop literal 'nan'
            return series.tolist(), c
    return [], None



def _parse_embedding_list_cell(cell) -> Optional[np.ndarray]:
    """
    Parse a cell that supposedly contains a JSON-like list of floats.
    Returns float32 ndarray or None if parsing fails.
    """
    if isinstance(cell, str):
        s = cell.strip()
        if s.startswith("[") or s.startswith("("):
            try:
                vec = json.loads(s.replace("(", "[").replace(")", "]"))
                return np.asarray(vec, dtype=np.float32)
            except Exception:
                pass
        # fallback: split by comma / whitespace
        sep = "," if "," in s else None
        try:
            parts = [float(x) for x in (s.split(sep) if sep else s.split()) if x != ""]
            return np.asarray(parts, dtype=np.float32)
        except Exception:
            return None
    elif isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=np.float32)
    elif cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    else:
        try:
            return np.asarray([float(cell)], dtype=np.float32)
        except Exception:
            return None


def _detect_embeddings(
    df: pd.DataFrame,
    embed_dim: int,
    use_embeddings: bool,
    ignore_embeddings: bool,
) -> np.ndarray:
    """
    Returns E with shape [N, embed_dim] (float32).
    Behavior:
      - if ignore_embeddings=True => zero context vectors of size embed_dim
      - else try wide columns 'emb_*' (preferred), then 'feat_*', then 'bit_*',
        then a single 'embedding' column (JSON-like list). If none found, fall back
        to zeros (but only if use_embeddings=True and embed_dim provided).
    """
    N = len(df)
    if ignore_embeddings or not use_embeddings:
        return np.zeros((N, embed_dim), dtype=np.float32)

    # Try common wide formats first
    for prefix in ("emb_", "feat_", "bit_"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            E = df[cols].to_numpy(dtype=np.float32)
            if E.shape[1] != embed_dim:
                raise ValueError(
                    f"embed_dim mismatch: CSV has {E.shape[1]} derived from '{prefix}*' "
                    f"but config/model expects {embed_dim}."
                )
            return E

    # Try a single JSON-like list column named 'embedding'
    if "embedding" in df.columns:
        vecs = []
        for cell in df["embedding"].values:
            arr = _parse_embedding_list_cell(cell)
            if arr is None:
                raise ValueError("Failed to parse some 'embedding' entries as a numeric list.")
            vecs.append(arr)
        E = np.stack(vecs).astype(np.float32)
        if E.shape[1] != embed_dim:
            raise ValueError(
                f"embed_dim mismatch: CSV 'embedding' lists have dim {E.shape[1]} "
                f"but config/model expects {embed_dim}."
            )
        return E

    # Nothing present → fall back to zeros (unconditional-like pretrain)
    return np.zeros((N, embed_dim), dtype=np.float32)

def collate_embeddings_batch(batch, feat_dim, dummy_idx, max_nodes_dataset):
    embed_batch, adj_batch, node_feat_batch, coord_batch = [], [], [], []
    label_batch, yield_batch, num_nodes_batch = [], [], []

    for (emb, adj, node_feats, n_atoms, label, yld, coords) in batch:
        emb = np.asarray(emb, dtype=np.float32)
        embed_batch.append(torch.from_numpy(emb))

        N = max_nodes_dataset

        # adjacency
        A = np.zeros((N, N), dtype=np.float32)
        if n_atoms > 0 and len(adj) >= n_atoms:
            a_np = np.asarray(adj, dtype=np.float32)
            A[:n_atoms, :n_atoms] = a_np[:n_atoms, :n_atoms]
        adj_batch.append(torch.from_numpy(A))

        # node features (pad + dummy)
        F = np.zeros((N, feat_dim), dtype=np.float32)
        for i in range(min(n_atoms, len(node_feats))):
            f = np.asarray(node_feats[i], dtype=np.float32)
            if f.shape[0] <= feat_dim:
                F[i, :f.shape[0]] = f
            else:
                F[i, :] = f[:feat_dim]
        if 0 <= dummy_idx < feat_dim:
            F[n_atoms:, dummy_idx] = 1.0
        node_feat_batch.append(torch.from_numpy(F))

        # coords
        C = np.zeros((N, 3), dtype=np.float32)
        if coords is not None and len(coords) > 0:
            c_np = np.asarray(coords, dtype=np.float32)
            m = min(N, c_np.shape[0])
            C[:m, :3] = c_np[:m, :3]
        coord_batch.append(torch.from_numpy(C))

        label_batch.append(label)
        yield_batch.append(yld)
        num_nodes_batch.append(n_atoms)

    return {
        "embeddings": torch.stack(embed_batch, 0),
        "adj": torch.stack(adj_batch, 0),
        "node_feats": torch.stack(node_feat_batch, 0),
        "coords": torch.stack(coord_batch, 0),
        "labels": label_batch,
        "yields": yield_batch,
        "num_nodes": num_nodes_batch,
    }


class EmbeddingDataset:
    """
    Versatile dataset for molecules with optional embeddings and optional labels/yields.

    Supported CSV schemas:
      1) SMILES only:               columns: ['smiles']
      2) SMILES + embeddings:       columns: ['smiles', 'emb_0'..'emb_{d-1}'] or single 'embedding'
      3) Full (optional):           columns: ['smiles', 'emb_*', 'label', 'Yield']

    Parameters
    ----------
    csv_file : str
        Path to CSV file.
    embed_dim : int
        Expected embedding dimension (required if no wide 'emb_*' columns and no 'embedding' column).
        Also used to size zero-context vectors when ignore_embeddings=True.
    use_embeddings : bool
        If False, embeddings are ignored and zeros are used (same as unconditional pretrain).
    ignore_embeddings : bool
        If True, force zeros even if embeddings exist (unconditional pretrain mode).
    atom_types : Optional[List[str]]
        Predefined atom-type vocabulary (include a dummy/stop token). If None, will infer.
    need_3d : bool
        If True, build 3D coordinates via RDKit ETKDG (heavy atoms only for graph).
    """

    def __init__(
        self,
        csv_file: str,
        embed_dim: int,
        use_embeddings: bool = True,
        ignore_embeddings: bool = False,
        atom_types: Optional[List[str]] = None,
        need_3d: bool = False,
    ):
        self.data = pd.read_csv(csv_file)

        # ---- Embeddings (robust detection) ----
        self.embeddings = _detect_embeddings(
            self.data, embed_dim=embed_dim, use_embeddings=use_embeddings, ignore_embeddings=ignore_embeddings
        )
        self.embed_dim = int(self.embeddings.shape[1])

        # ---- Other columns are OPTIONAL ----
        self.smiles_list, self.smiles_col = _get_smiles_list(self.data, preferred=None)
        if len(self.smiles_list) == 0:
            raise ValueError(
                "No SMILES column found in CSV. Expected one of: "
                f"{SMILES_CANDIDATES}. Training GraphVAE/GraphAF requires SMILES."
            )

        # not required for pretrain
        self.yields = self.data["Yield"].tolist() if "Yield" in self.data.columns else [None] * len(self.smiles_list)
        self.labels = self.data["label"].astype(str).tolist() if "label" in self.data.columns else ["unknown"] * len(self.smiles_list)

        # ---- Atom types vocab ----
        if atom_types is not None:
            self.atom_types = list(atom_types)
        else:
            atom_type_set = set()
            for smi in self.smiles_list:
                mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
                if mol:
                    for atom in mol.GetAtoms():
                        atom_type_set.add(atom.GetSymbol())
            atom_types_sorted = sorted(atom_type_set) if atom_type_set else ["C", "N", "O", "F", "S"]
            atom_types_sorted.append("Stop")  # dummy/stop token
            self.atom_types = atom_types_sorted

        # map → index
        self.atom_type_to_idx = {sym: idx for idx, sym in enumerate(self.atom_types)}
        if "Unknown" not in self.atom_type_to_idx:
            self.atom_type_to_idx["Unknown"] = len(self.atom_type_to_idx)
            self.atom_types.append("Unknown")
        self.dummy_idx = self.atom_type_to_idx.get("Stop", len(self.atom_types) - 1)
        self.num_types = len(self.atom_types)

        # ---- Graphs & optional 3D ----
        self.graphs: List[Tuple[List[List[int]], List[List[int]]]] = []
        self.coords_list: List[Optional[List[List[float]]]] = []
        self.num_atoms_list: List[int] = []
        self.max_nodes = 0

        for smi in self.smiles_list:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                n = 0
                adj = []
                node_feats = []
                coords = None
            else:
                adj, node_feats = chem_utils.get_adjacency_and_features(mol, self.atom_type_to_idx)
                n = mol.GetNumAtoms()
                coords = chem_utils.embed_molecule_3d(mol) if need_3d else None

            self.num_atoms_list.append(n)
            self.max_nodes = max(self.max_nodes, n)
            self.graphs.append((adj, node_feats))
            self.coords_list.append(coords)

    def __len__(self):
        return len(self.smiles_list) if self.smiles_list else len(self.embeddings)

    def __getitem__(self, idx):
        # emb ndarray → torch in collate (keep raw list/np to be consistent with original code)
        emb = self.embeddings[idx]
        adj, node_feats = self.graphs[idx]
        n_atoms = self.num_atoms_list[idx]
        label = self.labels[idx] if self.labels else None
        yld = self.yields[idx] if self.yields else None
        coords = self.coords_list[idx]
        return emb, adj, node_feats, n_atoms, label, yld, coords

    def collate_fn(self):
        # Return a picklable callable (no nested functions)
        return partial(
            collate_embeddings_batch,
            feat_dim=self.num_types,
            dummy_idx=self.dummy_idx,
            max_nodes_dataset=self.max_nodes,
        )

    # def collate_fn(self):
    #     """
    #     Collate a batch into padded tensors.
    #     Returns dict with keys:
    #       embeddings [B, D], adj [B, N, N], node_feats [B, N, T], coords [B, N, 3],
    #       labels (list[str]), yields (list[float|None]), num_nodes (list[int])
    #     """
    #     feat_dim = self.num_types
    #     dummy_idx = self.dummy_idx
    #     max_nodes_dataset = self.max_nodes

    #     def collate(batch):
    #         B = len(batch)
    #         embed_batch = []
    #         adj_batch = []
    #         node_feat_batch = []
    #         coord_batch = []
    #         label_batch = []
    #         yield_batch = []
    #         num_nodes_batch = []

    #         for (emb, adj, node_feats, n_atoms, label, yld, coords) in batch:
    #             # embeddings
    #             emb = np.asarray(emb, dtype=np.float32)
    #             embed_batch.append(torch.from_numpy(emb))

    #             # pad adjacency
    #             N = max_nodes_dataset
    #             A = np.zeros((N, N), dtype=np.float32)
    #             if n_atoms > 0 and len(adj) >= n_atoms:
    #                 a_np = np.asarray(adj, dtype=np.float32)
    #                 A[:n_atoms, :n_atoms] = a_np[:n_atoms, :n_atoms]
    #             adj_batch.append(torch.from_numpy(A))

    #             # pad node features to feat_dim
    #             F = np.zeros((N, feat_dim), dtype=np.float32)
    #             for i in range(min(n_atoms, len(node_feats))):
    #                 f = np.asarray(node_feats[i], dtype=np.float32)
    #                 if f.shape[0] <= feat_dim:
    #                     F[i, :f.shape[0]] = f
    #                 else:
    #                     F[i, :] = f[:feat_dim]
    #             # dummy for remaining atoms
    #             if 0 <= dummy_idx < feat_dim:
    #                 for i in range(n_atoms, N):
    #                     F[i, dummy_idx] = 1.0
    #             node_feat_batch.append(torch.from_numpy(F))

    #             # pad coords
    #             C = np.zeros((N, 3), dtype=np.float32)
    #             if coords is not None and len(coords) > 0:
    #                 c_np = np.asarray(coords, dtype=np.float32)
    #                 C[: min(N, c_np.shape[0]), :3] = c_np[: min(N, c_np.shape[0]), :3]
    #             coord_batch.append(torch.from_numpy(C))

    #             label_batch.append(label)
    #             yield_batch.append(yld)
    #             num_nodes_batch.append(n_atoms)

    #         return {
    #             "embeddings": torch.stack(embed_batch, dim=0),
    #             "adj": torch.stack(adj_batch, dim=0),
    #             "node_feats": torch.stack(node_feat_batch, dim=0),
    #             "coords": torch.stack(coord_batch, dim=0),
    #             "labels": label_batch,
    #             "yields": yield_batch,
    #             "num_nodes": num_nodes_batch,
    #         }

    #     return collate
