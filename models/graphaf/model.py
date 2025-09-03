import torch
import torch.nn as nn
from rdkit import Chem
from common import chem as chem_utils
from models.components.gnn_blocks import SimpleGraphEncoder
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAFGenerator(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, latent_dim, embed_dim, max_nodes=50):
        super(GraphAFGenerator, self).__init__()
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        # Networks for node and edge generation
        self.node_flow_net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128), nn.ReLU(),
            nn.Linear(128, node_feat_size)  # outputs logits for node type (including stop)
        )
        self.edge_flow_net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128), nn.ReLU(),
            nn.Linear(128, edge_feat_size)  # outputs logits/prob for edge existence or type
        )
        # Graph encoder to get representation of current partial graph
        self.graph_encoder = SimpleGraphEncoder(input_dim=node_feat_size, hidden_dim=embed_dim)
    
    def sample_graph(self, context_embedding):
        """
        Generate a molecular graph (as RDKit Mol) conditional on the given embedding.
        context_embedding: torch.Tensor of shape (1, embed_dim)
        """
        nodes = []        # list of node type indices
        mol = Chem.RWMol()
        # We'll generate nodes sequentially
        for t in range(self.max_nodes):
            # Compute graph representation of current partial graph
            if nodes:
                # Build temporary adjacency and node feature tensor for current graph
                n_current = len(nodes)
                # Create one-hot features for existing nodes
                node_feats = torch.zeros(n_current, self.node_feat_size)
                for i, type_idx in enumerate(nodes):
                    node_feats[i, type_idx] = 1.0
                # Adjacency for existing graph
                adj = torch.zeros(n_current, n_current)
                for i in range(n_current):
                    for j in range(n_current):
                        if mol.GetMol().GetBondBetweenAtoms(int(i), int(j)):
                            adj[i, j] = 1
                graph_repr = self.graph_encoder(node_feats, adj)
            else:
                # No nodes yet, use zero vector for graph_repr
                graph_repr = torch.zeros(self.embed_dim)
            # Sample a random latent
            z = torch.randn(1, self.latent_dim)
            # Combine embedding and graph representation
            context = torch.cat([z, context_embedding + graph_repr.view(1,-1)], dim=1)
            node_logits = self.node_flow_net(context).view(-1)  # shape (node_feat_size,)
            # Get probabilities and sample a node type
            node_prob = torch.softmax(node_logits, dim=0).detach().cpu().numpy()
            node_idx = int(torch.multinomial(torch.softmax(node_logits, dim=0), 1))
            # Check for stop token (assuming the last index corresponds to "Stop"/dummy)
            if node_idx == self.node_feat_size - 1:  # stop token
                break
            # Add the new atom to the molecule
            atom_symbol = "C"  # default to carbon; in a real scenario map index to actual element symbol
            mol_idx = mol.AddAtom(Chem.Atom(atom_symbol))
            nodes.append(node_idx)
            # Generate edges from this new node to all existing nodes except itself
            for j in range(mol_idx):
                edge_context = torch.cat([z, context_embedding + graph_repr.view(1,-1)], dim=1)
                edge_logit = self.edge_flow_net(edge_context).view(-1)  # outputs single logit if edge_feat_size=1
                edge_prob = torch.sigmoid(edge_logit).item()
                # Sample edge presence
                edge_flag = (edge_prob >= 0.5)
                if edge_flag:
                    try:
                        mol.AddBond(int(mol_idx), int(j), Chem.BondType.SINGLE)
                    except Exception as e:
                        # If adding bond fails (e.g., due to valence issues), skip it
                        continue
        # Sanitize and return molecule
        final_mol = mol.GetMol()
        try:
            Chem.SanitizeMol(final_mol)
        except Chem.SanitizeException:
            pass  # allow some invalid valences as outcome
        return final_mol
    
    def fit_step(self, batch):
        """
        Perform a training step (compute loss) on a batch of graphs.
        The batch is a dict with 'adj', 'node_feats', 'num_nodes', 'embeddings'.
        Uses teacher forcing to train the autoregressive model.
        """
        device = next(self.parameters()).device
        batch_loss = torch.tensor(0.0, device=device)

        B = len(batch['num_nodes'])
        for idx in range(B):
            n_nodes = int(batch['num_nodes'][idx])

            # Get actual adjacency and node features for this molecule from batch
            adj = batch['adj'][idx]                             # shape (max_nodes, max_nodes)
            node_feat_matrix = batch['node_feats'][idx]         # (max_nodes, node_feat_size)
            embed_vec = batch['embeddings'][idx].unsqueeze(0).to(device)  # (1, embed_dim)

            node_loss = torch.tensor(0.0, device=device)
            edge_loss = torch.tensor(0.0, device=device)

            # Initialize a partial graph representation (no nodes to start)
            nodes_added = []
            mol = Chem.RWMol()

            for t in range(n_nodes + 1):  # include stop step
                # --- Graph representation of current partial graph ---
                graph_repr = torch.zeros(self.embed_dim, device=device)
                if nodes_added:
                    cur_n = len(nodes_added)
                    # one-hot node features for partial graph
                    cur_node_feats = torch.zeros(cur_n, self.node_feat_size, device=device)
                    for i, type_idx in enumerate(nodes_added):
                        cur_node_feats[i, int(type_idx)] = 1.0

                    # adjacency from current RDKit mol
                    cur_adj = torch.zeros(cur_n, cur_n, dtype=torch.float32, device=device)
                    for i in range(cur_n):
                        for j in range(cur_n):
                            if mol.GetMol().GetBondBetweenAtoms(int(i), int(j)):
                                cur_adj[i, j] = 1.0

                    graph_repr = self.graph_encoder(cur_node_feats, cur_adj)  # -> (embed_dim,)
                    # ensure it's a flat vector
                    graph_repr = graph_repr.view(-1)

                # --- Context vector ---
                z = torch.zeros(1, self.latent_dim, device=device)  # zero latent during training
                # embed_vec: (1, embed_dim), graph_repr: (embed_dim,) -> (1, embed_dim)
                context = torch.cat([z, embed_vec + graph_repr.view(1, -1)], dim=1)

                # --- Node type prediction ---
                node_logits = self.node_flow_net(context).view(-1)  # [C_node]
                if t < n_nodes:
                    # target is actual node type at position t (fall back to STOP if out of range)
                    if node_feat_matrix.shape[0] > t:
                        target_type_idx = int(torch.argmax(node_feat_matrix[t]).item())
                    else:
                        target_type_idx = self.node_feat_size - 1  # STOP
                else:
                    target_type_idx = self.node_feat_size - 1  # STOP at final step

                node_target = torch.as_tensor([target_type_idx], dtype=torch.long, device=device)
                node_loss = node_loss + F.cross_entropy(node_logits.unsqueeze(0), node_target)

                # If STOP, end generation for this molecule
                if target_type_idx == self.node_feat_size - 1:
                    break

                # Otherwise, add this node to partial graph (default element placeholder)
                nodes_added.append(target_type_idx)
                try:
                    mol.AddAtom(Chem.Atom("C"))
                except Exception:
                    pass

                # --- Edge predictions from new node to existing ones ---
                # new node index = len(nodes_added)-1
                for j in range(len(nodes_added) - 1):
                    edge_logit = self.edge_flow_net(context).view(-1)  # shape [C_edge] or [1]
                    C_edge = edge_logit.numel()

                    # Target edge presence from ground-truth adjacency
                    # (t is the current "true" node index; j indexes previous added nodes)
                    # adj may be numpy/torch/list; current code used adj[t][j]
                    try:
                        gt = int(adj[t][j])
                    except Exception:
                        gt = int(adj[t, j].item() if torch.is_tensor(adj) else int(adj[t, j]))
                    has_edge = 1 if gt == 1 else 0

                    # ---- Loss selection to avoid size mismatch ----
                    if C_edge == 1:
                        # Binary logit → use BCEWithLogits with scalar target
                        target_edge = torch.as_tensor([float(has_edge)], device=edge_logit.device, dtype=edge_logit.dtype)
                        edge_loss = edge_loss + F.binary_cross_entropy_with_logits(edge_logit, target_edge)

                    elif C_edge == 2:
                        # Two-class (no-edge vs edge) → use CrossEntropy with class index {0,1}
                        target_class = torch.as_tensor([has_edge], dtype=torch.long, device=edge_logit.device)
                        edge_loss = edge_loss + F.cross_entropy(edge_logit.unsqueeze(0), target_class)

                    else:
                        # Multi-class (e.g., [no bond, single, double, triple])
                        # With binary adjacency, map: 0 -> NO_BOND (class 0), 1 -> SINGLE (class 1)
                        target_class_idx = 1 if has_edge == 1 else 0
                        target_class = torch.as_tensor([target_class_idx], dtype=torch.long, device=edge_logit.device)
                        edge_loss = edge_loss + F.cross_entropy(edge_logit.unsqueeze(0), target_class)

                        # Optionally add the bond to the mol only when we predict/know presence
                        if has_edge == 1:
                            try:
                                mol.AddBond(int(len(nodes_added) - 1), int(j), Chem.BondType.SINGLE)
                            except Exception:
                                pass

            # Sum losses for this molecule
            batch_loss = batch_loss + (node_loss + edge_loss)

        # Average loss over batch
        batch_loss = batch_loss / max(1, B)
        return batch_loss
