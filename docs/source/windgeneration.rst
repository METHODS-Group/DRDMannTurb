.. py:currentmodule:: drdmannturb

Wind Field Generation 
=====================

The other half of DRDMannTurb generates and visualizes 3D wind fields from spectra data. 

.. autoclass:: drdmannturb.GenerateFluctuationField 
    :members: 

Plotting Utilities 
------------------

.. currentmodule:: drdmannturb.wind_generation.wind_plot

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
