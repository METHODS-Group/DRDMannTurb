`DRDMannTurb`
=============

`DRDMannTurb` is an easy-to-use, data-driven (CPU and GPU (CUDA) capable) Python framework for
the automatic generation of synthetic turbulent wind fields, intended to be used by
wind engineers in their various applications domains. It is based off of
`Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_
and the associated previous Zenodo release.

The data-driven functionalities are implemented with `PyTorch <https://pytorch.org/docs/stable/index.html>`_.
The user will benefit from some familiarity with it; much of the `DRDMannTurb` API handles
``torch.Tensor``'s.

.. note::
   This project is still under development.

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
