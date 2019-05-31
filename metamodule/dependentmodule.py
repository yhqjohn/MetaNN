import torch
from torch.nn import Module

from copy import deepcopy
from collections import OrderedDict

from .utils import SubDict


class DependentModule(Module):
    def __new__(cls, module: Module):
        module = cls.to_dependentmodule(module)
        return module

    def __init__(self, *args, **kwargs):
        self._dependents = SubDict(self._buffers)
        self._dependents_shapes = {}

    def reinit(self):
        self._dependents = SubDict(self._buffers)
        self._dependents_shapes = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            value = self.to_dependentmodule(value)
        super(DependentModule, self).__setattr__(name, value)

    def register_dependent(self, name, tensor):
        if '_dependents' not in self.__dict__:
            raise AttributeError(
                "cannot assign dependent parameter before MetaModule.__init__() or MetaModule.reinit() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError("dependent parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("dependent parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("dependent parameter name can't be empty string \"\"")
        elif hasattr(self, name) and not name in self._dependents:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("cannot assign '{}' object to dependent parameter '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(tensor), name))
        else:
            self._dependents[name] = tensor
            if tensor is not None:
                self._dependents_shapes[name] = tensor.shape
            else:
                if name not in self._dependents_shapes.keys():
                    self._dependents_shapes[name] = None

    def named_dependents(self, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = (lambda module: module._dependents.items())(module)
            for k, v in members:
                if v in memo and v is not None:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def dependents(self, recurse=True):
        for name, param in self.named_dependents(recurse=recurse):
            yield param

    def update_shapes(self):
        def gen():
            for name, value in self._dependents.items():
                if value is None:
                    if name in self._dependents_shapes:
                        yield name, self._dependents_shapes[name]
                else:
                    yield name, value.shape

        self._dependents_shapes = dict(gen())

    def _substitute(self, name, value):
        if name not in self._dependents:
            raise KeyError("{} is not in dependent parameters".format(name))
        elif self._dependents_shapes[name] is not None and self._dependents_shapes[name] != value.shape:
            raise ValueError("size mismatch for {}, expect {}, got{}".format(
                name, self._dependents_shapes[name], value.shape))

        self._dependents[name] = value

    def _substitute_from_params_dict(self, params_dict, prefix, strict=True):
        for name in self._dependents:
            key = prefix + name
            if strict:
                if key in params_dict:
                    self._substitute(name, params_dict[key])
                else:
                    raise ValueError("params_dict and interim parameters mismatch")
            else:
                if key in params_dict:
                    try:
                        self._substitute(name, params_dict[key])
                    except (KeyError, ValueError):
                        pass

    def substitute(self, named_params, strict=True):
        params_dict = dict(named_params)

        def load(module: DependentModule, prefix='', _strict=True):
            module._substitute_from_params_dict(params_dict, prefix, strict=_strict)
            for name, child in module._modules.items():
                load(child, prefix+name+'.', _strict=_strict)

        load(self, _strict=strict)

    def substitute_from_list(self, params):
        named_params = ((k, v) for (k, _), v in zip(self.named_dependents(), params))
        self.substitute(named_params)

    def clear_params(self, init=False, clear_filter=lambda x: True):

        def clear_fn(module: DependentModule):
            if clear_filter(module):
                for name, value in module._parameters.items():
                    module._dependents[name] = torch.Tensor(value)
                module._parameters = OrderedDict()
                module.update_shapes()

            if not init:
                for key in module._dependents:
                    module._dependents[key] = None

        self.apply(clear_fn)
        return self

    @classmethod
    def _make_subclass(cls, module: Module):
        if not isinstance(module, cls):
            module.__class__ = type("Meta"+type(module).__name__, (cls, type(module)), {})
            module.reinit()
        return module

    @classmethod
    def to_dependentmodule(cls, module: Module, recurse=True):
        if not recurse:
            module = cls._make_subclass(module)
        else:
            module.apply(lambda x: cls.to_dependentmodule(x, recurse=False))
        return module

    @classmethod
    def stateless(cls, module: Module, clear_filter=lambda x: True):
        return cls.to_dependentmodule(deepcopy(module)).clear_params(clear_filter)