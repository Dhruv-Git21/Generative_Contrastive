import torch
import torch.nn as nn
from models.components.equivariant_blocks import EquivariantGNN

class EquivariantFlowMatchingModel(nn.Module):
    def __init__(self, embed_dim, num_atom_types=5):
        super(EquivariantFlowMatchingModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_atom_types = num_atom_types
        # Equivariant coordinate network
        self.coord_net = EquivariantGNN(input_dim=num_atom_types, hidden_dim=128)
        # Type velocity network
        self.type_net = nn.Sequential(
            nn.Linear(128 + embed_dim, 64), nn.SiLU(),
            nn.Linear(64, num_atom_types)
        )
    def forward(self, atom_positions, atom_features, context_embed):
        """
        Compute coordinate and type velocity given current state and context embedding.
        atom_positions: tensor (N,3)
        atom_features: tensor (N, num_atom_types) one-hot type features
        context_embed: tensor (1, embed_dim)
        """
        N = atom_positions.size(0)
        # Get latent feature per atom from equivariant GNN
        coord_feat = self.coord_net(atom_positions, atom_features)  # (N, hidden_dim)
        # Incorporate global context for type update
        context_expand = context_embed.expand(N, -1)  # (N, embed_dim)
        combined_feat = torch.cat([coord_feat, context_expand], dim=1)  # (N, 128+embed_dim)
        type_velocity = self.type_net(combined_feat)  # (N, num_atom_types)
        # Compute coordinate velocity: sum of differences weighted by learned function
        # We'll compute pairwise weights using coord_feat and positions
        diff = atom_positions.view(N,1,3) - atom_positions.view(1,N,3)  # (N,N,3)
        dist = torch.norm(diff, dim=-1)  # (N,N)
        # Use coord_feat to compute weights
        w = torch.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # If either atom is dummy (identified by all-zero feature vector or type index maybe), skip
                # Here assume dummy type is index 0 and features for dummy atom would be one-hot at 0.
                if atom_features[i,0] == 1 or atom_features[j,0] == 1:
                    w_ij = 0.0
                else:
                    # Simple weight: use inverse distance squared times a learned scalar from features
                    inv_dist = 1.0 / (dist[i,j] + 1e-6)
                    # Combine features i and j (e.g., dot product or norm of sum)
                    feat_pair = (coord_feat[i] + coord_feat[j]).norm().item()
                    w_ij = inv_dist * feat_pair
                w[i,j] = w_ij
        # Normalize weights for stability
        # Compute coordinate velocity for each atom
        coord_velocity = torch.zeros_like(atom_positions)
        for i in range(N):
            # Sum contributions from all other atoms
            force = torch.zeros(3)
            for j in range(N):
                if i == j: 
                    continue
                force += w[i,j] * (atom_positions[j] - atom_positions[i])
            coord_velocity[i] = force
        return coord_velocity, type_velocity
    
    def sample(self, context_embed, num_atoms=10, steps=50, step_size=0.1):
        """
        Generate a molecule by simulating flow matching from random initial state to final state.
        context_embed: tensor (1, embed_dim)
        """
        # Initialize positions randomly (e.g., gaussian)
        atom_positions = torch.randn((num_atoms, 3)) * (self.coord_net.hidden_dim**0.5)
        # Initialize type features as all dummy (assuming index 0 is dummy type)
        atom_features = torch.zeros((num_atoms, self.num_atom_types))
        atom_features[:, 0] = 1.0  # all start as dummy
        # Integrate ODE with Euler steps
        for t in range(steps):
            coord_vel, type_vel = self.forward(atom_positions, atom_features, context_embed)
            atom_positions = atom_positions + step_size * coord_vel
            atom_features = atom_features + step_size * type_vel
            # No normalization of atom_features to remain in logits space
        # At end, determine final types
        final_types = torch.argmax(atom_features, dim=1)  # index of highest logit
        # Filter out dummy atoms
        atoms = []
        for i, t in enumerate(final_types):
            if t.item() == 0:
                continue  # dummy
            # Map type index to element symbol (basic mapping for demo)
            symbol_map = {1: 'C', 2: 'N', 3: 'O', 4: 'F'}
            atom_symbol = symbol_map.get(t.item(), 'C')
            pos = atom_positions[i].detach().cpu().numpy()
            atoms.append((atom_symbol, pos[0], pos[1], pos[2]))
        return atoms
    
    def fit_step(self, batch):
        # Expect batch with 'coords', 'node_feats', 'embeddings', 'num_nodes'
        B = len(batch['num_nodes'])
        total_loss = 0.0
        for i in range(B):
            n = batch['num_nodes'][i]
            coords = batch['coords'][i][:n]
            node_feats = batch['node_feats'][i][:n]
            embed = batch['embeddings'][i].unsqueeze(0)
            # Set initial positions random and initial features to dummy for all atoms
            init_positions = torch.randn_like(coords)
            init_features = torch.zeros_like(node_feats)
            # Mark all initial as dummy (index 0)
            if init_features.shape[1] > 0:
                init_features[:,0] = 1.0
            # Final positions and features (target)
            final_positions = coords
            final_features = node_feats
            # Simulate simple linear path for training
            t = 0.5  # pick a midpoint
            current_positions = (1-t)*init_positions + t*final_positions
            current_features = (1-t)*init_features + t*final_features
            # Compute velocities
            coord_vel_pred, type_vel_pred = self.forward(current_positions, current_features, embed)
            # Target velocities (linear path => constant velocity = final - initial)
            coord_vel_target = (final_positions - init_positions)
            type_vel_target = (final_features - init_features)
            # Mask dummy atoms (target dummy have no change)
            # Actually, we'll include them, expecting model to output ~0 for them
            # Compute MSE losses
            coord_loss = torch.mean((coord_vel_pred - coord_vel_target)**2)
            type_loss = torch.mean((type_vel_pred - type_vel_target)**2)
            total_loss += (coord_loss + type_loss)
        return total_loss / B
