Field Generation API
===================

The field generation module provides tools for generating synthetic turbulent wind fields.

FluctuationFieldGenerator
------------------------

Main interface for generating synthetic turbulent fields.

.. autoclass:: drdmannturb.fluctuation_generation.FluctuationFieldGenerator
   :members:
   :undoc-members:
   :show-inheritance:

   **Supported Models:**

   - ``"DRD"``: Deep Rapid Distortion model (requires pre-trained model)
   - ``"Mann"``: Classical Mann model
   - ``"VK"``: Von Karman model

   **Example Usage:**

   .. code-block:: python

      from drdmannturb.fluctuation_generation import FluctuationFieldGenerator

      # Create generator
      generator = FluctuationFieldGenerator(
          friction_velocity=0.5,
          reference_height=100.0,
          grid_dimensions=[1000.0, 500.0, 300.0],
          grid_levels=[6, 6, 5],
          model="DRD",
          path_to_parameters="calibrated_model.pkl"
      )

      # Generate field
      field = generator.generate()

Gaussian Random Fields
---------------------

Core algorithms for generating Gaussian random fields.

.. automodule:: drdmannturb.fluctuation_generation.gaussian_random_fields
   :members:
   :undoc-members:

   **Key Classes:**

   - ``VectorGaussianRandomField``: Main field generator
   - ``CovarianceKernel``: Base class for covariance kernels
   - ``VonKarmanCovariance``: Von Karman covariance kernel
   - ``MannCovariance``: Mann covariance kernel
   - ``NNCovariance``: Neural network covariance kernel

Sampling Methods
---------------

Different sampling strategies for field generation.

.. automodule:: drdmannturb.fluctuation_generation.sampling_methods
   :members:
   :undoc-members:

   **Available Methods:**

   - ``vf_fftw``: Vector FFT-based sampling (recommended)
   - ``vf_fft``: Vector FFT sampling
   - ``vf_fft_parallel``: Parallel vector FFT sampling

Low Frequency Generation
-----------------------

Tools for generating low-frequency components.

.. automodule:: drdmannturb.fluctuation_generation.low_frequency
   :members:
   :undoc-members:

   **Key Classes:**

   - ``LowFreqGenerator``: Low-frequency field generator
   - ``FluctuationFieldGenerator``: Low-frequency field generator

Plotting Utilities
-----------------

Visualization tools for generated fields.

.. automodule:: drdmannturb.fluctuation_generation.wind_plot
   :members:
   :undoc-members:
