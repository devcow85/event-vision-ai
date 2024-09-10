import random

import numpy as np
import torch


def set_seed(random_seed):
    """reproducible option

    Args:
        random_seed (int): seed value
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
