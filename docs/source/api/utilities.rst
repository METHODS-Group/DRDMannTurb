Utilities API
=============

Utility functions and tools for DRDMannTurb.

Data Generation
---------------

Functions for generating synthetic data for testing and validation.

.. automodule:: drdmannturb.spectra_fitting.data_generator
   :members:
   :undoc-members:
   :no-index:

   **Functions:**

   - ``generate_von_karman_spectra()``: Generate Von Karman spectra
   - ``generate_kaimal_spectra()``: Generate Kaimal spectra

Plotting Utilities
------------------

Visualization tools for generated wind fields, built on Plotly.

.. automodule:: drdmannturb.fluctuation_generation.wind_plot
   :members:
   :undoc-members:
   :no-index:

   **Functions:**

   - ``create_grid()``: Build the evaluation grid for a field
   - ``format_wind_field()``: Format a raw field for plotting
   - ``plot_velocity_components()``: Plot each velocity component separately
   - ``plot_velocity_magnitude()``: Plot the velocity magnitude

Configuration
-------------

Configuration and setup utilities.

.. automodule:: drdmannturb.parameters
   :members:
   :undoc-members:
   :no-index:

   **Classes:**

   - ``IntegrationParameters``: Numerical integration settings
   - ``LossParameters``: Loss function configuration
