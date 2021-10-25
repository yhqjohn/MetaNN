MetaNN for PyTorch Meta Learning
=====================================

.. image:: https://readthedocs.org/projects/metann/badge/?version=latest
    :target: https://metann.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

1. Introduction
____________________

In meta learner scenario, it is common use dependent variables as parameters, and back propagate the gradient of the parameters. However, parameters of PyTorch Module are designed to be leaf nodes and it is forbidden for parameters to have grad_fn. Meta learning coders are therefore forced to rewrite the basic layers to adapt the meta learning requirements.

This module provide an extension of torch.nn.Module, DependentModule that has dependent parameters, allowing the differentiable dependent parameters. It also provide the method to transform nn.Module into DependentModule, and turning all of the parameters of a nn.Module into dependent parameters.

2. Installation
__________________

.. code-block:: python

    pip install MetaNN

3. Example
___________

PyTorch suggest all parameters of a module to be independent variables. Using DependentModule arbitrary torch.nn.module can be transformed into dependent module.

.. code-block:: python

    from metann import DependentModule
    from torch import nn
    net = torch.nn.Sequential(
        nn.Linear(10, 100),
        nn.Linear(100, 5))
    net = DependentModule(net)
    print(net)

Higher-level api such as MAML class are more recommended to use.

.. code-block:: python

    from metann.meta import MAML, default_evaluator_classification as evaluator
    from torch import nn
    net = torch.nn.Sequential(
        nn.Linear(10, 100),
        nn.Linear(100, 5))
    )
    maml = MAML(net, steps_train=5, steps_eval=10, lr=0.01)
    output = maml(data_train)
    loss = evaluator(output, data_test)
    loss.backward()


4. Documents
_____________

The documents are available at ReadTheDocs.
`MetaNN <https://metann.readthedocs.io/>`__

5. License
__________

`MIT <http://opensource.org/licenses/MIT>`__

Copyright (c) 2019-present, Hanqiao Yu