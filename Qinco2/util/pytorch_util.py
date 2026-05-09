import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader


def get_dataloader(X, y):
    X_tensor, y_tensor = torch.from_numpy(X), torch.from_numpy(y)
    return DataLoader(TensorDataset(X_tensor, y_tensor), shuffle=True)


class LogisticRegressionModel(nn.Module):
    def __init__(self, ndim_input):
        super(LogisticRegressionModel, self).__init__()
        self.linear_layer = nn.Linear(ndim_input, 1)

    def forward(self, X_tensor):
        return torch.sigmoid(self.linear_layer(X_tensor))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False