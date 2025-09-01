import time
import logging
import torch

def get_device():
    """Return the default device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Timer:
    """Simple timer context manager for measuring code execution time."""
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start

def log_epoch_metrics(epoch, train_loss, val_loss=None):
    """Log metrics for an epoch."""
    if val_loss is not None:
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    else:
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
