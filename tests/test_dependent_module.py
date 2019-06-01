import pytest
import torch
from torch import nn
from metann import DependentModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test_dependent_module():
    net = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.Conv2d(3, 3, 3),
        Flatten(),
        nn.Linear(3, 4),
    )
    net = DependentModule(net).to(device)
    x = torch.randn(3, 3, 5, 5).to(device)
    print(net)
    assert net(x).shape == torch.Size([3, 4])
