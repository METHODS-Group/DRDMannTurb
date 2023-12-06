.. py:currentmodule:: drdmannturb

Visualizing Training
====================

``DRDMannTurb`` relies on TensorBoard for visualizing training results including the behavior of all involved loss terms. Please refer to the TensorBoard API for specifics as well as the :py:class:`LossAggregator` arguments for how to set the logging directory and changing metadata.

We also provide a simple utility for plotting out the training progress from a single TensorBoard log. 

.. currentmodule:: drdmannturb.common 

.. autofunction:: plot_loss_logs
