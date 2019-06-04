from torch.nn import Module
from copy import deepcopy

from .dependentmodule import DependentModule


class Learner(Module):
    r"""
    This module extends nn.Module by providing functional method.
    :param module: a nn.Module module
    """
    def __init__(self, module: Module):
        super(Learner, self).__init__()
        self.module = module
        self.stateless = DependentModule.stateless(deepcopy(module))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def functional(self, params, training, *args, **kwargs):
        r"""
        :param iterable params:  input model parameters for functional
        :param training: if the functional set to trainning=True
        :param args: input
        :param kwargs: input
        :return: return the output of model

        Examples::

            >>>learner = Learner(net)
            >>>outputs = learner.functional(net.parameters(), training=True, x)

        """
        self.stateless.substitute_from_list(params)
        if training:
            self.stateless.train()
        else:
            self.stateless.eval()
        return self.stateless(*args, **kwargs)

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)