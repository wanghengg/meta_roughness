import torch
import torch.nn as nn
import torch.nn.functional as F

class RoughnessNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, params=None):
        if params is None:
            return self.net(x)

        x = F.linear(x, params['net.0.weight'], params['net.0.bias'])
        x = F.relu(x)
        x = F.linear(x, params['net.2.weight'], params['net.2.bias'])
        x = F.relu(x)
        x = F.linear(x, params['net.4.weight'], params['net.4.bias'])
        return x
