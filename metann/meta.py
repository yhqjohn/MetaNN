import torch
from torch import nn
from .utils.containers import DefaultList
from metann import DependentModule, ProtoModule
import numpy as np


is_tensor = np.vectorize(lambda x: isinstance(x, torch.Tensor))


def default_evaluator_classification(model, data, criterion=nn.CrossEntropyLoss()):
    x, y = data
    logits = model(x)
    loss = criterion(logits, y)
    return loss


class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()

    def forward(self, *args, inplace=False, **kwargs):
        if inplace:
            return self.forward_pure(*args, **kwargs)
        else:
            return self.forward_inplace(*args, **kwargs)

    def forward_pure(self, model, data):
        raise NotImplementedError

    def forward_inplace(self, model, data):
        raise NotImplementedError


class GDLearner(Learner):
    def __init__(self, steps, lr, create_graph=True, evaluator=default_evaluator_classification):
        super(GDLearner, self).__init__()
        self.steps = steps
        self.sgd = SequentialGDLearner(lr, momentum=0, create_graph=create_graph, evaluator=evaluator)

    def forward(self, model, data, retain_graph=True, **kwargs):
        kwargs['model'] = model
        kwargs['data'] = [data, ]*self.steps
        kwargs['retain_graph'] = retain_graph
        return self.sgd(**kwargs)


class SequentialGDLearner(Learner):
    def __init__(self, lr, momentum=0.5, create_graph=True, evaluator=default_evaluator_classification):
        super(SequentialGDLearner, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.create_graph = create_graph
        self.evaluator = evaluator

    def forward_pure(self, model, data, evaluator=None):
        evaluator = self.evaluator if evaluator is None else evaluator
        model = ProtoModule(model)
        model.train()
        fast_weights = np.array(list(model.parameters()))
        actives = is_tensor(fast_weights)
        for batch in data:
            fast_loss = evaluator(model.functional(fast_weights), batch)
            grads = torch.autograd.grad(fast_loss, fast_weights[actives],
                                        create_graph=self.create_graph)
            fast_weights[actives] = [w - self.lr * g for (w, g) in zip(fast_weights[actives], grads)]
        return model.functional(fast_weights)

    def forward_inplace(self, model, data, evaluator=None):
        evaluator = self.evaluator if evaluator is None else evaluator
        optim = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        for batch in data:
            optim.zero_grad()
            loss = evaluator(model, batch)
            loss.backward()
            optim.step()
        return model


class MAML(nn.Module):
    def __init__(self, model, steps_train, steps_eval, lr,
                 evaluator=default_evaluator_classification, first_order = False):
        super(MAML, self).__init__()
        self.model = model
        self.steps_train = steps_train
        self.steps_eval = steps_eval
        self.lr = lr
        self.evaluator = evaluator
        self.first_order = first_order

    def forward(self, data):
        if self.training:
            steps = self.steps_train
        else:
            steps = self.steps_eval
        learner = GDLearner(self.steps_train, self.lr, create_graph=not self.first_order)
        return learner(self.model, data, evaluator=self.evaluator)
