import torch
from torch import nn
from torch.optim import SGD
from metann.meta import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test_gdlearner():
    model0 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.Conv2d(3, 3, 3),
        Flatten(),
        nn.Linear(3, 4),
    ).to(device)
    x = torch.randn(3, 3, 5, 5).to(device)
    y = torch.randint(0, 4, (3,)).to(device)
    data = (x, y)
    learner = GDLearner(20, 0.1)

    model1 = learner(model0, data)
    loss = default_evaluator_classification(model1, data)
    assert loss <= 0.2
    print(loss)
    loss = default_evaluator_classification(model0, data)
    print(loss)
    model1 = learner(model0, data, retain_graph=False)
    loss = default_evaluator_classification(model0, data)
    print(loss)
    assert loss <= 0.2

    model0 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.Conv2d(3, 3, 3),
        Flatten(),
        nn.Linear(3, 4),
    ).to(device)

    x = torch.randn(3, 3, 5, 5).to(device)
    dummy_x = x.clone().requires_grad_()
    y = torch.randint(0, 4, (3,)).to(device)
    data = (x, y)
    dummy_data = (dummy_x, y)
    model1 = learner(model0, dummy_data)
    loss = default_evaluator_classification(model1, data)
    loss.backward()
    print(dummy_x.grad.abs().sum().item())
    assert dummy_x.grad.abs().sum().item() > 0
