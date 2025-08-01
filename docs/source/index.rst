`DRDMannTurb`
=============

`DRDMannTurb` (Deep Rapid Distortion Mann Turbulence) is a GPU-accelerated, data-driven Python framework for
(1) the calibration of spectral tensor models of wind turbulence to field data and (2) the generation of synthetic turbulent wind fields
based on these calibrated models. This package is intended to be used by wind engineers in applications requiring rapid
simulation of realistic wind turbulence. It is based off of the Deep Rapid Distortion models
presented in `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_. The data-driven functionalities are
GPU-accelerated via `PyTorch <https://pytorch.org/docs/stable/index.html>`_.

.. figure:: ../../drd.gif

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started

.. raw:: html

    <hr>

.. toctree::
   :maxdepth: 2

   examples

.. raw:: html

    <hr>

.. toctree::
   :maxdepth: 2

   forward_api

.. raw:: html

    <hr>

.. toctree::
   :maxdepth: 2

   internal_api

.. raw:: html

    <hr>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
