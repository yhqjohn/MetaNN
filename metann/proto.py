from torch.nn import Module
from copy import deepcopy

from .dependentmodule import DependentModule


class ProtoModule(Module):
    r"""
    This module extends nn.Module by providing functional method.
    It is a stateful module, but allows you to call its stateless functional
    :param module: a nn.Module module
    """
    def __init__(self, module: Module):
        super(ProtoModule, self).__init__()
        self.module = module
        self.stateless = DependentModule.stateless(deepcopy(module))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def functional(self, params, training=None):
        r"""
        :param iterable params:  input model parameters for functional
        :param training: if the functional set to trainning=True
        :return: return the output of model

        Examples::

            >>>learner = Learner(net)
            >>>outputs = learner.functional(net.parameters(), training=True)(x)

        """
        training = self.training if training is None else training
        self.stateless.substitute_from_list(params)
        if training:
            self.stateless.train()
        else:
            self.stateless.eval()
        return self.stateless

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
