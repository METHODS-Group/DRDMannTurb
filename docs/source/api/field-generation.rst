Field Generation API
====================

The field generation module provides tools for generating synthetic turbulent wind fields.

FluctuationFieldGenerator
-------------------------

Main interface for generating synthetic turbulent fields.

.. autoclass:: drdmannturb.fluctuation_generation.FluctuationFieldGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   The generator is driven by a :ref:`covariance kernel <api-covariance-kernels>`, which determines
   the turbulence model: ``VonKarmanCovariance``, ``MannCovariance``, or ``NNCovariance`` (built
   from a trained DRD model).

   **Example Usage:**

   .. code-block:: python

      import numpy as np

      from drdmannturb.fluctuation_generation import FluctuationFieldGenerator, MannCovariance

      covariance = MannCovariance(length_scale=50.0, E0=1.0, Gamma=3.9)

      generator = FluctuationFieldGenerator(
          grid_dimensions=np.array([1000.0, 500.0, 300.0]),
          grid_levels=np.array([6, 6, 5]),
          covariance=covariance,
      )

      field = generator.generate(num_blocks=2, zref=100.0, uref=10.0, z0=0.1, windprofiletype="LOG")

.. _api-covariance-kernels:

Covariance Kernels
------------------

Covariance kernels evaluate the square-root of the spectral tensor used to correlate Gaussian noise.

.. automodule:: drdmannturb.fluctuation_generation.covariance_kernels
   :members:
   :undoc-members:

.. autoclass:: drdmannturb.fluctuation_generation.NNCovariance
   :members:
   :undoc-members:
   :show-inheritance:

Gaussian Random Fields
----------------------

Core algorithms for generating Gaussian random fields.

.. automodule:: drdmannturb.fluctuation_generation.gaussian_random_fields
   :members:
   :undoc-members:

Sampling Methods
----------------

Different sampling strategies for correlating the noise with the covariance kernel, selected by
name: ``"fft"``, ``"fftw"``, ``"vf_fftw"`` (vector field FFTW; used by ``FluctuationFieldGenerator``),
``"dst"``, and ``"dct"``.

.. automodule:: drdmannturb.fluctuation_generation.sampling_methods
   :members:
   :undoc-members:

Low Frequency Generation
------------------------

The 2D low-frequency model of Syed & Mann (2024), used for the 2D+3D model.

.. automodule:: drdmannturb.fluctuation_generation.low_frequency.fluctuation_field_generator
   :members:
   :undoc-members:

Plotting Utilities
------------------

Visualization tools for generated fields.

.. automodule:: drdmannturb.fluctuation_generation.wind_plot
   :members:
   :undoc-members:
