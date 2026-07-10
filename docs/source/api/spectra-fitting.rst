Spectra Fitting API
===================

The spectra fitting module provides tools for calibrating spectral tensor models to wind turbulence data.

CalibrationProblem
------------------

The main interface for calibrating spectral tensor models to spectral data.

.. autoclass:: drdmannturb.spectra_fitting.CalibrationProblem
   :members:
   :undoc-members:
   :show-inheritance:

   **Key Methods:**

   - ``calibrate()``: Run the calibration process
   - ``plot()``: Plot the calibrated model against the data
   - ``save_model()``: Save the calibrated model (reload with ``SpectralTensorModel.load_model``)

   **Example Usage:**

   .. code-block:: python

      import torch

      from drdmannturb.nn_modules import TauNet
      from drdmannturb.spectra_fitting import CalibrationProblem
      from drdmannturb.spectra_fitting.spectral_tensor_models import (
          RDT_SpectralTensor,
          TauNet_ELT,
          VonKarman_ESM,
      )

      # Compose the spectral tensor model
      model = RDT_SpectralTensor(
          eddy_lifetime_model=TauNet_ELT(taunet=TauNet(n_layers=2, hidden_layer_sizes=[10, 10])),
          energy_spectrum_model=VonKarman_ESM(),
          L_init=50.0,
          gamma_init=3.9,
          sigma_init=0.05,
      )

      # Set up and run the calibration
      problem = CalibrationProblem(model=model, data=your_spectral_data)
      problem.calibrate(
          optimizer_class=torch.optim.LBFGS,
          lr=1.0,
          max_epochs=100,
      )

      # Persist the trained model
      checkpoint_path = problem.save_model()

CustomDataLoader
----------------

Handles loading and formatting of spectral data from various sources.

.. autoclass:: drdmannturb.spectra_fitting.CustomDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

   **Data Format:**

   The CSV file should contain columns:
   - ``freq``: Frequency values
   - ``uu``: u-component auto-spectra (required)
   - ``vv``: v-component auto-spectra (required)
   - ``ww``: w-component auto-spectra (required)
   - ``uw``: u-w cross-spectra (optional)
   - ``vw``: v-w cross-spectra (optional)
   - ``uv``: u-v cross-spectra (optional)

   **Example Usage:**

   .. code-block:: python

      from drdmannturb.spectra_fitting import CustomDataLoader

      # Load data from CSV file
      data_loader = CustomDataLoader("spectra_data.csv")
      data = data_loader.format_data()

      # Use in calibration
      problem = CalibrationProblem(data=data, ...)

Spectral Tensor Models
----------------------

Base classes and implementations for spectral tensor models.

.. automodule:: drdmannturb.spectra_fitting.spectral_tensor_models
   :members:
   :undoc-members:

   **Available Models:**

   - ``EddyLifetimeModel``: Base class for eddy lifetime models
   - ``TauNet_ELT``: Neural network-based eddy lifetime model
   - ``Mann_ELT``: Classical Mann eddy lifetime model
   - ``TwoThirds_ELT``: Two-thirds power law model
   - ``Constant_ELT``: Constant eddy lifetime model
   - ``EnergySpectrumModel``: Base class for energy spectrum models
   - ``VonKarman_ESM``: Von Karman energy spectrum model
   - ``Learnable_ESM``: Learnable energy spectrum model
   - ``RDT_SpectralTensor``: Rapid Distortion Theory spectral tensor

Loss Functions
--------------

Loss functions and aggregators for model calibration.

.. automodule:: drdmannturb.spectra_fitting.loss_functions
   :members:
   :undoc-members:

   **Key Components:**

   - ``LossAggregator``: Combines the loss terms (log-space MSE, first/second-order penalties,
     network regularization, and optionally a spectral coherence MSE), weighted by ``LossParameters``.

Data Generation
---------------

Utilities for generating synthetic spectral data.

.. automodule:: drdmannturb.spectra_fitting.data_generator
   :members:
   :undoc-members:

   **Functions:**

   - ``generate_von_karman_spectra()``: Generate Von Karman spectra
   - ``generate_kaimal_spectra()``: Generate Kaimal spectra

One-Point Spectra
-----------------

Core functionality for computing one-point spectra.

.. automodule:: drdmannturb.spectra_fitting.one_point_spectra
   :members:
   :undoc-members:
