import torch
import numpy as np
import random

def init_random_seed(random_set=42):
    torch.manual_seed(random_set)
    torch.cuda.manual_seed(random_set)
    np.random.seed(random_set)
    random.seed(random_set)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count