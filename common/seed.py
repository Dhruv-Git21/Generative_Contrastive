import random
import numpy as np
import torch

def seed_all(seed: int):
    """Set random seeds for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optionally set deterministic flags if needed (could slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
