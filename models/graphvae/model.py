import torch
import torch.nn as nn
from models.components.gnn_blocks import SimpleGraphEncoder

def reparameterize(mu, logvar):
    """Sample from N(mu, sigma^2) via reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class GraphVAE(nn.Module):
    def __init__(self, max_nodes, node_feat_dim, embed_dim, latent_dim):
        super(GraphVAE, self).__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        # Encoder
        self.encoder_gnn = SimpleGraphEncoder(input_dim=node_feat_dim, hidden_dim=128)
        self.encoder_mean = nn.Linear(128 + embed_dim, latent_dim)
        self.encoder_logvar = nn.Linear(128 + embed_dim, latent_dim)
        # Decoder
        output_dim = max_nodes * max_nodes + max_nodes * node_feat_dim
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def encode(self, node_feats, adj, embed):
        # node_feats: (N, node_feat_dim), adj: (N,N), embed: (embed_dim,)
        graph_repr = self.encoder_gnn(node_feats, adj)  # 128-dim graph embedding
        combined = torch.cat([graph_repr, embed], dim=-1)
        mu = self.encoder_mean(combined)
        logvar = self.encoder_logvar(combined)
        return mu, logvar
    
    def decode(self, z, embed):
        # z: (batch, latent_dim), embed: (batch, embed_dim)
        x = torch.cat([z, embed], dim=-1)
        out = self.decoder_net(x)
        # Split output into adjacency and node features
        batch_size = out.shape[0]
        # First part: adj (max_nodes*max_nodes)
        adj_flat = out[:, :self.max_nodes*self.max_nodes]
        node_feat_flat = out[:, self.max_nodes*self.max_nodes:]
        # Reshape
        adj_logits = adj_flat.view(batch_size, self.max_nodes, self.max_nodes)
        node_feat_logits = node_feat_flat.view(batch_size, self.max_nodes, self.node_feat_dim)
        return adj_logits, node_feat_logits
    
    def forward(self, node_feats, adj, embed):
        mu, logvar = self.encode(node_feats, adj, embed)
        z = reparameterize(mu, logvar)
        adj_logits, node_logits = self.decode(z.unsqueeze(0), embed.unsqueeze(0))
        return adj_logits, node_logits, mu, logvar
    
    def fit_step(self, batch):
        # We assume batch contains a single graph for simplicity (or we handle one by one similarly to GraphAF).
        # For vectorized training, code can be extended.
        embeddings = batch['embeddings']  # shape (B, embed_dim)
        adjs = batch['adj']              # shape (B, max_nodes, max_nodes)
        node_feats = batch['node_feats'] # shape (B, max_nodes, node_feat_dim)
        B = embeddings.size(0)
        recon_loss_total = 0.0
        kld_loss_total = 0.0
        for i in range(B):
            n = batch['num_nodes'][i]
            embed = embeddings[i]
            # Robustly slice adjacency matrix
            adj_full = adjs[i]
            # Always slice to [max_nodes, max_nodes] first
            adj_full = adj_full[:self.max_nodes, :self.max_nodes]
            # Then slice to [n, n]
            adj = adj_full[:n, :n]
            # Assert shape for debugging
            assert adj.shape == (n, n), f"adj shape mismatch: got {adj.shape}, expected ({n}, {n})"
            node_feat_full = node_feats[i][:self.max_nodes, :]
            node_feat = node_feat_full[:n, :]
            mu, logvar = self.encode(node_feat, adj, embed)
            z = reparameterize(mu, logvar)
            adj_logits, node_logits = self.decode(z.unsqueeze(0), embed.unsqueeze(0))
            # Get predicted and target for loss
            adj_pred = adj_logits.view(self.max_nodes, self.max_nodes)
            node_pred = node_logits.view(self.max_nodes, self.node_feat_dim)
            # Build target matrices (with padding for dummy nodes)
            target_adj = torch.zeros_like(adj_pred)
            target_node = torch.zeros_like(node_pred)
            # Fill actual adjacency
            target_adj[:n, :n] = adj
            # Fill actual node one-hot
            target_node[:n, :] = node_feat
            # Compute reconstruction loss
            # For adjacency, use binary cross entropy on each entry
            recon_adj_loss = nn.functional.binary_cross_entropy_with_logits(adj_pred, target_adj)
            # For node features, use cross entropy for each node's feature vector
            # Convert target_node (one-hot) to indices
            target_indices = torch.argmax(target_node, dim=-1)
            # Only consider first n (actual nodes) plus treat others as dummy
            # Actually include all, because for padded ones target index will presumably be dummy index (0 if all zeros)
            recon_node_loss = 0.0
            for j in range(self.max_nodes):
                # Skip if target is all zeros (which means dummy), we will consider that as dummy index
                tgt_idx = int(target_indices[j].item())
                logits = node_pred[j].unsqueeze(0)
                recon_node_loss += nn.functional.cross_entropy(
                    logits, torch.tensor([tgt_idx], device=logits.device)
                )
            recon_node_loss = recon_node_loss / self.max_nodes
            recon_loss = recon_adj_loss + recon_node_loss
            # KLD loss
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            recon_loss_total += recon_loss
            kld_loss_total += kld
        recon_loss_avg = recon_loss_total / B
        kld_loss_avg = kld_loss_total / B
        beta = 1.0  # weight for KLD, could be tuned
        total_loss = recon_loss_avg + beta * kld_loss_avg
        return total_loss
