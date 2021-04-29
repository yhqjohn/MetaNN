import pytest
import sys
sys.path.append('../')

import torch
from torch import nn
from torch.optim import SGD
from metann import ProtoModule, tensor_copy, mimo_functional
from metann.meta import default_evaluator_classification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test_proto_module():
    net = ProtoModule(
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
        outs = net.functional(params, True)(x)
        loss = criterion(outs, y)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            params = [(a-0.01*b).requires_grad_() for a, b in zip(params, grads)]
    print(loss)
    assert loss <= 0.05


def test_proto_module_mimo():
    net = ProtoModule(
        nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.Conv2d(3, 3, 3),
            Flatten(),
            nn.Linear(3, 4),
        )
    ).to(device)
    net.train()
    x = torch.randn(3, 3, 5, 5).to(device)
    y = torch.randint(0, 4, (3,)).to(device)
    data = [(x, y), ]*4
    evaluator = default_evaluator_classification
    params = list(net.parameters())
    for i in range(500):
        params_lst = [tensor_copy(params) for _ in range(4)]
        outs = mimo_functional(net, params_lst)(data, [evaluator]*4)
        loss = sum(outs)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            params = [(a-0.01*b).requires_grad_() for a, b in zip(params, grads)]
    assert loss <= 0.05
