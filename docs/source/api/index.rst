API Reference
=============

This section provides comprehensive API documentation for DRDMannTurb.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   spectra-fitting
   field-generation
   models
   utilities

Overview
--------

DRDMannTurb provides several key modules:

**Spectra Fitting** (`drdmannturb.spectra_fitting`)
   Core functionality for calibrating spectral tensor models to wind turbulence data.

**Field Generation** (`drdmannturb.fluctuation_generation`)
   Tools for generating synthetic turbulent wind fields using calibrated models.

**Neural Networks** (`drdmannturb.nn_modules`)
   Neural network architectures for learning eddy lifetime functions.

**Parameters** (`drdmannturb.parameters`)
   Parameter classes for configuring models and calibration processes.

**Utilities** (`drdmannturb.spectra_fitting.data_generator`, etc.)
   Helper functions for data generation, plotting, and configuration.

Quick Reference
---------------

**Main Classes:**

- ``CalibrationProblem``: Main interface for model calibration
- ``FluctuationFieldGenerator``: Main interface for field generation
- ``TauNet``: Neural network for eddy lifetime function
- ``CustomDataLoader``: Data loading and formatting
- ``RDT_SpectralTensor``: Rapid Distortion Theory spectral tensor

**Key Functions:**

- ``generate_von_karman_spectra()``: Generate synthetic Von Karman spectra
- ``generate_kaimal_spectra()``: Generate synthetic Kaimal spectra

**Parameter Classes:**

- ``PhysicalParameters``: Physical model parameters
- ``LossParameters``: Loss function configuration
- ``IntegrationParameters``: Numerical integration settings
