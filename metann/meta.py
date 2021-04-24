import torch
from torch import nn
from .utils.containers import DefaultList, MultipleList
from .utils.numpy import to_array
from metann import DependentModule, ProtoModule
from metann.utils.containers import DefaultList
import numpy as np


is_tensor = np.vectorize(lambda x: isinstance(x, torch.Tensor))


def active_indices(lst):
    indices = []
    for k, v in enumerate(lst):
        if isinstance(v, torch.Tensor):
            indices.append(k)
    return indices


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
            return self.forward_inplace(*args, **kwargs)
        else:
            return self.forward_pure(*args, **kwargs)

    def forward_pure(self, model, data):
        raise NotImplementedError

    def forward_inplace(self, model, data):
        raise NotImplementedError


class GDLearner(Learner):
    def __init__(self, steps, lr, create_graph=True, evaluator=default_evaluator_classification):
        super(GDLearner, self).__init__()
        self.steps = steps
        self.sgd = SequentialGDLearner(lr, momentum=0, create_graph=create_graph, evaluator=evaluator)

    def forward(self, model, data, inplace=False, **kwargs):
        kwargs['model'] = model
        kwargs['data'] = [data, ]*self.steps
        kwargs['inplace'] = inplace
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
        fast_weights = MultipleList(list(model.parameters()))
        velocities = DefaultList(lambda: 0)
        actives = active_indices(fast_weights)
        for batch in data:
            fast_loss = evaluator(model.functional(fast_weights), batch)
            grads = torch.autograd.grad(fast_loss, fast_weights[actives],
                                        create_graph=self.create_graph)
            velocities = [grad + velocity*self.momentum for (grad, velocity) in zip(grads, velocities)]
            fast_weights[actives] = [w - self.lr * g for (w, g) in zip(fast_weights[actives], velocities)]
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


def _rms_prop(data, grad, state):
    if state['r'] is None:
        state['r'] = torch.zeros_like(data)
    if state['centered'] and state['grad_avg'] is None:
        state['grad_avg'] = torch.zeros_like(data)
    alpha, eps = state['alpha'], state['eps']

    r = state['r'] = state['r'].mul(alpha).addcmul(1-alpha, grad, grad)

    if state['centered']:
        grad_avg = state['grad_avg'] = state['grad_avg'].mul(alpha).add(1 - alpha, grad)
        avg = r.addcmul(-1, grad_avg, grad_avg).sqrt().add(eps)
    else:
        avg = r.sqrt().add(eps)

    return data.addcdiv(-state['lr'], grad, avg), state


class RMSPropLearner(Learner):
    def __init__(self, lr=1e-2, alpha=0.99, eps=1e-8, centered=False, create_graph=True,
                 evaluator=default_evaluator_classification, steps=None):
        super(RMSPropLearner, self).__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self.evaluator = evaluator
        self.steps = steps
        self.create_graph = create_graph

    def forward_pure(self, model, data, evaluator=None):
        if self.steps is not None:
            data = [data, ]*self.steps
        evaluator = self.evaluator if evaluator is None else evaluator
        model = ProtoModule(model)
        model.train()
        fast_weights = MultipleList(list(model.parameters()))
        actives = active_indices(fast_weights)
        states = DefaultList(lambda: {'centered': self.centered, 'alpha': self.alpha,
                                      'lr': self.lr, 'eps': self.eps,
                                      'r': None, 'grad_avg': None})
        for batch in data:
            fast_loss = evaluator(model.functional(fast_weights), batch)
            grads = torch.autograd.grad(fast_loss, fast_weights[actives],
                                        create_graph=self.create_graph)
            _fast_weights = []
            for i, (w, g, s) in enumerate(zip(fast_weights[actives], grads, states)):
                w, states[i] = _rms_prop(w, g, s)
                _fast_weights.append(w)
            fast_weights[actives] = _fast_weights
        return model.functional(fast_weights)

    def forward_inplace(self, model, data, evaluator=None):
        if self.steps is not None:
            data = [data, ]*self.steps
        evaluator = self.evaluator if evaluator is None else evaluator
        optim = torch.optim.RMSprop(model.parameters(), lr=self.lr,
                                    alpha=self.alpha, eps=self.eps, centered=self.centered)
        for batch in data:
            optim.zero_grad()
            loss = evaluator(model, batch)
            loss.backward()
            optim.step()
        return model


class MAML(nn.Module):
    def __init__(self, model, steps_train, steps_eval, lr,
                 evaluator=default_evaluator_classification, first_order=False):
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
