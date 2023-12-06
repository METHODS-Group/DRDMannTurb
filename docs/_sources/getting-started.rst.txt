.. py:currentmodule:: drdmannturb

Getting Started
===============

`DRDMannTurb` consists of two major components: fitting a Deep Rapid Distortion (DRD) model to spectra that are either generated from the Mann model or taken from real-world wind observation, and generating divergence-free fluctuation fields that have similar spectra characteristics. Additional utilities for data pre-processing, such as interpolation and filtering, as well as 3D visualization tools are provided as part of the package. 

Please consider reading `the original DRD paper <https://arxiv.org/pdf/2107.11046.pdf>`_ and the examples. Please also see the UML diagrams for both :doc:`spectra fitting <./uml_drd>` and :doc:`fluctuation field generation <./uml_fluct_gen>` to see how the models are configured.  

Spectra Fitting 
---------------

DRD models perform an operator regression task that characterizes an optimal nonlocal covariance kernel via neural network training. These kernels are part of a family  that preserve fluid flow properties such as mass conservation as well as turbulence statistics up to second order. The package is based on `the Mann model <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/spatial-structure-of-neutral-atmospheric-surfacelayer-turbulence/ACFE1EA8C45763481CBEB193B314E2EB>`_, which admits three parameters that
may also determined directly from data by DRD models: the Kolmogorov constant multiplied by the two-thirds power law for the rate of viscous dissipation, a turbulence length scale, and a non-dimensional time-scale related to the eddy-lifetime function. ``DRDMannTurb`` allows users to easily fit the Mann model to turbulence characteristics by also approximating the function related to the lifetime of the eddies through deep neural networks. These networks have a wide range of customizability in terms of architecture and their training is facilitated via ``PyTorch``. 

The overall workflow consists of a ``OnePointSpectraDataGenerator`` which provides the data over which a DRD model is to be trained. This class also provides pre-processing utilities for interpolation of data to a common wave vector domain and filtering the curves to improve fits with noisy data. Alternatively, data can be generated from the Mann model. This enables the functionality of performing pure Mann model fits to data and for generating fluctuation fields based on the Mann model. 

Field Generation
----------------

Fluctuation field generation involves one class and is performed block-wise: rather than generating the field in the whole domain, it is partitioned into several blocks that are then composed together to form a single uncorrelated block of turbulence. Please see the final section of the original DRD paper for a more in-depth discussion. Plotting utilities facilitated by ``Plotly`` are provided. 