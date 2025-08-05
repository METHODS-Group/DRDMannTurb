DRDMannTurb
=============

``DRDMannTurb`` (Deep Rapid Distortion Mann Turbulence) is a data-driven Python framework for
(1) the calibration of spectral tensor models of wind turbulence to field data and (2) the generation of synthetic turbulent wind fields
based on these calibrated models. This package is intended for use by wind engineers in applications requiring rapid
simulation of realistic wind turbulence. It is based off of the Deep Rapid Distortion models
presented in `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_.

.. figure:: ../../drd.gif

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   getting-started
   user-guide/installation
   user-guide/quickstart
   user-guide/spectra-fitting
   user-guide/field-generation
   user-guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index
   api/spectra-fitting
   api/field-generation
   api/models
   api/utilities

.. toctree::
   :maxdepth: 2
   :caption: Development:

   development/contributing
   development/changelog
   development/citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
