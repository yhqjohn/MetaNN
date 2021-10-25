import torch
from torch.nn import Module
from copy import deepcopy

from .dependentmodule import DependentModule
from .utils.containers import MultipleList


class ProtoModule(Module):
    r"""
    This module extends nn.Module by providing functional method.
    It is a stateful module, but allows you to call its stateless functional.

    Args:
        module: a nn.Module module
    """
    def __init__(self, module: Module):
        super(ProtoModule, self).__init__()
        self.module = module
        self.stateless = DependentModule.stateless(deepcopy(module))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def functional(self, params, training=None):
        r"""
        Args:
            iterable params:  input model parameters for functional
            training: if the functional set to trainning=True
        Returns:
            return the output of model

        Examples:
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


def mimo_functional(proto: ProtoModule, params_lsts):
    def mimo_foward(inputs_lst, eval_lst):
        output_lst = []
        for params, input, evaluator in zip(params_lsts, inputs_lst, eval_lst):
            evaluator = (lambda x, y: x(y)) if evaluator is None else evaluator
            out = evaluator(proto.functional(params), input)
            output_lst.append(out)
        return output_lst

    return mimo_foward


def tensor_copy(tensor_lst):
    if isinstance(tensor_lst, torch.Tensor):
        return tensor_lst.clone()
    elif tensor_lst is None:
        return tensor_lst
    else:
        return MultipleList([tensor_copy(i) for i in tensor_lst])
