import pytest
import torch
from torch import nn
from torch.optim import SGD
from metann import Learner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test_learner():
    net = Learner(
        nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.Conv2d(3, 3, 3),
            Flatten(),
            nn.Linear(3, 4),
        )
    ).to(device)
    x = torch.randn(3, 3, 5, 5).to(device)
    y = torch.randint(0, 4, (3,)).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(net.parameters())
    for i in range(500):
        outs = net.functional(params, True, x)
        loss = criterion(outs, y)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            params = [(a-0.01*b).requires_grad_() for a, b in zip(params, grads)]
    print(loss)
    assert loss <= 0.05
