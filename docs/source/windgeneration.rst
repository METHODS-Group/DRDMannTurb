.. py:currentmodule:: drdmannturb

Fluctuation Field Generation 
============================

DRDMannTurb provides generic methods for generating fluctuation fields from specified spectra models. Please refer to :doc:`the UML diagram <./uml_fluct_gen>` to see the class hierarchy. 

.. autoclass:: drdmannturb.GenerateFluctuationField 
    :members: 

Plotting Utilities 
------------------

.. currentmodule:: drdmannturb.fluctuation_generation.wind_plot

.. autofunction:: plot_velocity_magnitude

.. autofunction:: plot_velocity_components 

.. autofunction:: format_wind_field 

.. autofunction:: create_grid

Gaussian Random Fields
----------------------

.. toctree::
   :maxdepth: 2
   
   grf

Covariance Kernels
------------------

.. toctree::
   :maxdepth: 2

   cov_kernels

Sampling Methods
----------------

.. toctree::
   :maxdepth: 1

   sampling
