from torch.nn import Module
from copy import deepcopy

from .dependentmodule import DependentModule


class Learner(Module):
    def __init__(self, module: Module):
        super(Learner, self).__init__()
        self.module = module
        self.stateless = DependentModule.stateless(deepcopy(module))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def functional(self, params, training, *args, **kwargs):
        self.stateless.substitute_from_list(params)
        if training:
            self.stateless.train()
        else:
            self.stateless.eval()
        return self.stateless(*args, **kwargs)

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)