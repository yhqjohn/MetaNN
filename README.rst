MetaNN for PyTorch Meta Learning
=====================================

1. Introduction
____________________

In meta learner scenario, it is common use dependent variables as parameters, and back propagate the gradient of the parameters. However, parameters of PyTorch Module are designed to be leaf nodes and it is forbidden for parameters to have grad_fn. Meta learning coders are therefore forced to rewrite the basic layers to adapt the meta learning requirements.

This module provide an extension of torch.nn.Module, DependentModule that has dependent parameters, allowing the differentiable dependent parameters. It also provide the method to transform nn.Module into DependentModule, and turning all of the parameters of a nn.Module into dependent parameters.

2. Installation
__________________

.. code-block::

    pip install MetaNN

3. Example
___________

.. code-block::

    from metann import DependentModule, Learner
    from torch import nn
    net = torch.nn.Sequential(
        nn.Linear(10, 100),
        nn.Linear(100, 5))
    net = DependentModule(net)
    print(net)

I suggest you to use higher-level api such as MAML class.

.. code-block::

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

`MetaNN <https://metann.readthedocs.io/>`__

This won't build correctly with the heavy dependency PyTorch, so I updated the sphinx built html to GitHub. I hate to use mock to solve This problem, I suggest you to clone the repository and view the html docs yourself.

5. License
__________

`MIT <http://opensource.org/licenses/MIT>`__

Copyright (c) 2019-present, Hanqiao Yu