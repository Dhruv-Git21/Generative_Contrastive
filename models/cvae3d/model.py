import torch
import torch.nn as nn

class LigandGenerator3D(nn.Module):
    def __init__(self, embed_dim, latent_dim, grid_size=24, num_atom_types=5):
        super(LigandGenerator3D, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_atom_types = num_atom_types
        # Fully connect to initial 3x3x3x256 grid
        self.fc = nn.Linear(latent_dim + embed_dim, 256 * 3 * 3 * 3)
        # Transposed convolutional decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(64, num_atom_types, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, z, embed):
        x = torch.cat([z, embed], dim=-1)  # shape: (batch, latent_dim+embed_dim)
        init_feats = self.fc(x)           # (batch, 256*3*3*3)
        init_feats = init_feats.view(-1, 256, 3, 3, 3)
        density_grid = self.decoder_conv(init_feats)  # (batch, num_atom_types, grid_size, grid_size, grid_size)
        return density_grid
    def fit_step(self, batch):
        # Expect batch contains 'coords' (B, max_nodes, 3), 'node_feats' (B, max_nodes, num_types), and 'embeddings'
        embeddings = batch['embeddings']
        coords = batch['coords']
        node_feats = batch['node_feats']
        B = embeddings.size(0)
        # Generate target density grid from coords and node types
        # We will accumulate loss for each sample
        total_loss = 0.0
        for i in range(B):
            embed = embeddings[i].unsqueeze(0)
            coord_list = coords[i]
            node_feat_mat = node_feats[i]  # includes dummy atoms as one-hot
            # Create target grid for this molecule
            target_grid = torch.zeros((1, self.num_atom_types, self.grid_size, self.grid_size, self.grid_size))
            # Determine origin for grid (center)
            origin = (self.grid_size - 1) / 2.0
            # Loop through real atoms (exclude dummy where type one-hot is all zeros or dummy type)
            for j in range(coord_list.shape[0]):
                # Determine type index
                type_idx = int(torch.argmax(node_feat_mat[j]).item())
                # Skip dummy type
                if type_idx >= self.num_atom_types:
                    continue
                # Get coordinates
                x, y, z = coord_list[j]
                # If coordinate is 0,0,0 and dummy, skip
                if type_idx == 0 and x.item() == 0.0 and y.item() == 0.0 and z.item() == 0.0:
                    # Assuming dummy type is index 0 for simplicity here
                    continue
                # Map coordinate to nearest grid index
                gx = int(round(x.item() + origin))
                gy = int(round(y.item() + origin))
                gz = int(round(z.item() + origin))
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size and 0 <= gz < self.grid_size:
                    target_grid[0, type_idx, gx, gy, gz] = 1.0
            # Forward pass with latent set to zero (deterministic)
            z = torch.zeros((1, self.latent_dim))
            pred_grid = self.forward(z, embed)
            # Loss: use binary cross-entropy on each voxel for each channel
            loss = nn.functional.binary_cross_entropy_with_logits(pred_grid, target_grid)
            total_loss += loss
        return total_loss / B
