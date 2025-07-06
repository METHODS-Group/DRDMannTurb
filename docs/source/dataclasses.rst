.. py:currentmodule:: drdmannturb

Dataclass Primitives
====================

``DRDMannTurb`` operates with four fundamental data classes that configure the DRD model to be used, physical parameters of the scenario, training data generation, and loss functions. Please refer to :doc:`this UML diagram <./uml_drd>` to see how these are composed to inform the :py:class:`CalibrationProblem` and :py:class:`OnePointSpectraDataGenerator` classes.

Neural Network Parameters
-------------------------

.. autoclass:: drdmannturb.NNParameters
    :members:


Physical Parameters
-------------------

.. autoclass:: drdmannturb.PhysicalParameters
    :members:


Problem/Training Parameters
---------------------------

.. autoclass:: drdmannturb.ProblemParameters
    :members:

Loss Function Parameters
------------------------

Also refer to :py:class:`LossAggregator` for loss function evaluation from these parameters. The loss function for spectra fitting is determined by the following minimization problem:

.. math::
    \min _{\boldsymbol{\theta}}\left\{\operatorname{MSE}[\boldsymbol{\theta}]+\alpha \operatorname{Pen}[\boldsymbol{\theta}]+\beta \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]\right\}

where the loss terms are defined as follows:

.. math::
    \operatorname{MSE}[\boldsymbol{\theta}]:=\frac{1}{L} \sum_{i=1}^4 \sum_{j=1}^L\left(\log \left|J_i\left(f_j\right)\right|-\log \left|\widetilde{J}_i\left(f_j, \boldsymbol{\theta}\right)\right|\right)^2,

where :math:`L` is the number of data points :math:`f_j \in \mathcal{D}`. The data :math:`J_i (f_j)` is evaluated using the Kaimal spectra. The second-order penalization term is defined as

.. math::
    \operatorname{Pen}_2[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|} \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial^2 \log \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\left(\partial \log k_1\right)^2}\right)\right\|_{\mathcal{D}}^2,

and the first-order penalization term is defined as

.. math::
    \operatorname{Pen}_1[\boldsymbol{\theta}]:=\frac{1}{|\mathcal{D}|} \sum_{i=1}^4\left\|\operatorname{ReLU}\left(\frac{\partial \log \left|\widetilde{J}_i(\cdot, \boldsymbol{\theta})\right|}{\partial \log k_1}\right)\right\|_{\mathcal{D}}^2.

Note that the norm :math:`||\cdot||_{\mathcal{D}}` is defined as

.. math::
    \|g\|_{\mathcal{D}}^2:=\int_{\mathcal{D}}|g(f)|^2 \mathrm{~d}(\log f).

The regularization term is defined as

.. math::
    \operatorname{Reg}\left[\boldsymbol{\theta}_{\mathrm{NN}}\right]:=\frac{1}{N} \sum_{i=1}^N \theta_{\mathrm{NN}, i}^2.


The LossParameters class correspondingly sets the weights of these terms.

.. autoclass:: drdmannturb.LossParameters
    :members:
