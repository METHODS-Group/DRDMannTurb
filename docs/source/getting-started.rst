.. py:currentmodule:: drdmannturb

Getting Started
===============

`DRDMannTurb` consists of two major components: (1) fitting a Deep Rapid Distortion (DRD) spectral tensor model to one-point spectra from real-world wind observations, and (2) generating divergence-free fluctuation fields from the inferred spectral tensor using a novel and efficient block-wise approach. Additional utilities for data pre-processing, such as interpolation and filtering, as well as 3D visualization tools are provided as part of the package. 

Please consider reading `the original DRD paper <https://arxiv.org/pdf/2107.11046.pdf>`_ and the examples. Please also see the UML diagrams for both :doc:`spectra fitting <./uml_drd>` and :doc:`fluctuation field generation <./uml_fluct_gen>` to see how the models are configured.  

Installation
=============

The following sections will guide you through the installtion of DRDMannTurb and its dependencies. The following instructions should work on any operating system (OS) that is supported by `Anaconda`_, including in particular: **Windows**, **macOS**, and **Linux**. 

#. Clone DRDMannTurb from its `GitHub repository <https://github.com/METHODS-Group/DRDMannTurb>`_.
   Enter the directory and check out the branch of your choice.
   The latest development version will be available under the branch ``develop``.

   .. code-block:: shell

      git clone https://github.com/METHODS-Group/DRDMannTurb.git
      cd DRDMannTurb
      git checkout main

#. Create an Anaconda environment called ``drdmannturb_env`` for installing DRDMannTurb.
   Use the default environment specs in ``env_drdmannturb.yml`` to create it.
   Then activate the environment:

   .. code-block:: shell

      conda env create -n drdmannturb_env -f requirements/env_drdmannturb.yml
      conda activate drdmannturb_env

#. Install the local DRDMannTurb source files as Python package using ``pip``:

   .. code-block:: shell

      python -m pip install -e ./

Spectra Fitting 
---------------

``DRDMannTurb`` performs an operator regression task that characterizes an optimal nonlocal covariance kernel via neural network training. These kernels are part of a family of DRD models that preserve physical fluid flow properties such as mass conservation as well as turbulence statistics up to second order. DRD models are generalizations of `the Mann model <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/spatial-structure-of-neutral-atmospheric-surfacelayer-turbulence/ACFE1EA8C45763481CBEB193B314E2EB>`_, which admits three parameters: the Kolmogorov constant multiplied by the two-thirds power law for the rate of viscous dissipation, a turbulence length scale, and a non-dimensional time-scale related to the eddy-lifetime function. ``DRDMannTurb`` allows users to easily extend the classical Mann model to a DRD model by using neural networks to approximate the function related to the lifetime of the eddies. These networks have a wide range of customizability in terms of architecture and their training is facilitated via ``PyTorch``.  This leads to a model that can better fit field data without changing the complexity of generating the associated synthetic turbulence wind field.

The overall workflow consists of a ``OnePointSpectraDataGenerator`` which provides the data over which a DRD model is to be trained. This class also provides pre-processing utilities for interpolation of data to a common wave vector domain and filtering the curves to improve fits with noisy data. Alternatively, data can be generated from the Mann model. This enables the functionality of performing pure Mann model fits to data and for generating fluctuation fields based on the Mann model.

Field Generation
----------------

Fluctuation field generation involves one class and is performed in an efficient block-wise manner: rather than generating the field in the whole domain, the field is generated locally on overlapping blocks that partition the domain and are then composed together to form a single turbulence field. Please see the final section of the original DRD paper for a more in-depth discussion. Plotting utilities facilitated by ``Plotly`` are provided. 
