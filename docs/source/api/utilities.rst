Utilities API
=============

Utility functions and tools for DRDMannTurb.

Data Generation
--------------

Functions for generating synthetic data for testing and validation.

.. automodule:: drdmannturb.spectra_fitting.data_generator
   :members:
   :undoc-members:

   **Functions:**

   - ``generate_von_karman_spectra()``: Generate Von Karman spectra
   - ``generate_kaimal_spectra()``: Generate Kaimal spectra

Plotting Utilities
-----------------

Visualization tools for spectra and fields.

.. automodule:: drdmannturb.fluctuation_generation.wind_plot
   :members:
   :undoc-members:

   **Functions:**

   - ``plot_spectra()``: Plot spectral components
   - ``plot_field()``: Plot generated fields
   - ``plot_coherence()``: Plot coherence functions

Covariance Utilities
-------------------

Tools for working with covariance kernels.

.. automodule:: drdmannturb.fluctuation_generation.cmap_util
   :members:
   :undoc-members:

   **Functions:**

   - ``compute_covariance()``: Compute covariance matrices
   - ``plot_covariance()``: Visualize covariance structure

Configuration
------------

Configuration and setup utilities.

.. automodule:: drdmannturb.parameters
   :members:
   :undoc-members:

   **Classes:**

   - ``IntegrationParameters``: Numerical integration settings
   - ``LossParameters``: Loss function configuration
   - ``ProblemParameters``: Problem-specific settings
