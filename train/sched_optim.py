import torch

def make_optimizer(params, config):
    opt_name = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    if opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        # Default to Adam
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, config):
    sched_name = config.get('scheduler', None)
    if not sched_name:
        return None
    sched_name = sched_name.lower()
    if sched_name == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == 'multistep':
        milestones = config.get('milestones', [10, 20])
        gamma = config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif sched_name == 'cosine':
        T_max = config.get('T_max', 50)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        return None
