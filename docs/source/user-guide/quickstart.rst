Quickstart Guide
================

This guide provides a base example for using ``DRDMannTurb`` to first calibrate a spectral tensor
model to provided data and then generate a synthetic turbulent wind field using that calibrated
model. See Examples 2 and 5 for the complete, commented versions of both halves.

Basic Spectra Fitting
---------------------

.. code-block:: python

   import torch
   import drdmannturb as drdmt
   from drdmannturb.spectra_fitting import CalibrationProblem
   from drdmannturb.spectra_fitting import spectral_tensor_models as stm

   zref, ustar = 90.0, 0.556

   # Synthetic Kaimal spectra as training data (swap in a CustomDataLoader for real data)
   k1 = torch.logspace(-1, 2, 20, dtype=torch.float64) / zref
   data = drdmt.spectra_fitting.data_generator.generate_kaimal_spectra(k1=k1, zref=zref, ustar=ustar)

   # A DRD model: neural-network eddy lifetime + Von Karman energy spectrum
   model = stm.RDT_SpectralTensor(
       eddy_lifetime_model=stm.TauNet_ELT(taunet=drdmt.TauNet(n_layers=2, hidden_layer_sizes=[10, 10])),
       energy_spectrum_model=stm.VonKarman_ESM(),
       L_init=0.59 * zref,
       gamma_init=3.9,
       sigma_init=3.2 * ustar**2 / zref ** (2 / 3),
   )

   pb = CalibrationProblem(model=model, data=data)
   pb.calibrate(optimizer_class=torch.optim.LBFGS, lr=1.0, max_epochs=20)

   checkpoint = pb.save_model()  # writes a .pt checkpoint

Basic Field Generation
----------------------

.. code-block:: python

   import numpy as np
   from drdmannturb.fluctuation_generation import FluctuationFieldGenerator, NNCovariance
   from drdmannturb.spectra_fitting import spectral_tensor_models as stm

   drd_model = stm.RDT_SpectralTensor.load_model(checkpoint)
   scales = drd_model.calibration_scales

   covariance = NNCovariance(
       spectral_tensor_model=drd_model,
       length_scale=zref / (2 * np.pi * scales["freq_scale"]),
       amplitude=ustar**2 * scales["spectral_scale"],
   )

   generator = FluctuationFieldGenerator(
       grid_dimensions=np.array([300.0, 150.0, 150.0]),
       grid_levels=np.array([6, 5, 5]),
       covariance=covariance,
   )

   field = generator.generate(num_blocks=2, zref=zref, uref=11.4, z0=0.02, windprofiletype="LOG")
   generator.save_to_vtk("./fluctuation_field")
