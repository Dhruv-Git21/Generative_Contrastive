import logging
import torch
from train import sched_optim

def train_model(model, train_loader, config, val_loader=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """Unified training loop for the model."""
    model.to(device)
    model.train()
    # Create optimizer and scheduler
    optim = sched_optim.make_optimizer(model.parameters(), config)
    sched = sched_optim.make_scheduler(optim, config)
    epochs = config.get('epochs', 10)
    log_interval = config.get('log_interval', 1)
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in train_loader:
            # Move data to device
            batch = move_batch_to_device(batch, device)
            optim.zero_grad()
            loss = model.fit_step(batch)
            # If the model returns multiple values (e.g., loss and others), assume first is loss
            if isinstance(loss, tuple) or isinstance(loss, list):
                loss_val = loss[0]
            else:
                loss_val = loss
            loss_val.backward()
            optim.step()
            total_loss += loss_val.item()
        avg_loss = total_loss / len(train_loader)
        if epoch % log_interval == 0:
            if val_loader:
                val_loss = evaluate_model(model, val_loader, device)
                logging.info(f"Epoch {epoch}/{epochs}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                logging.info(f"Epoch {epoch}/{epochs}: Train Loss = {avg_loss:.4f}")
        if sched:
            sched.step()
    # Save model checkpoint if path specified
    save_path = config.get('checkpoint', None)
    if save_path:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model checkpoint saved to {save_path}")

def evaluate_model(model, data_loader, device=torch.device('cpu')):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            loss = model.fit_step(batch)
            if isinstance(loss, tuple) or isinstance(loss, list):
                loss = loss[0]
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader)

def move_batch_to_device(batch, device):
    """Move all tensor contents of batch to the given device."""
    for key, val in batch.items():
        if torch.is_tensor(val):
            batch[key] = val.to(device)
    return batch
