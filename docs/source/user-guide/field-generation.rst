Field Generation
================

The ``fluctuation_generation`` sub-module generates synthetic turbulent wind fields with the "Mann
method": Gaussian noise is correlated in Fourier space with the square-root of a spectral tensor,
and the field is built block-by-block in the streamwise direction with shared noise in the overlap
regions.

Covariance kernels
------------------

The turbulence model is selected through a covariance kernel, which evaluates the square-root of
the spectral tensor:

- ``VonKarmanCovariance``: the isotropic Von Karman model;
- ``MannCovariance``: the Mann (1994) uniform-shear model;
- ``NNCovariance``: a trained DRD model (see :doc:`spectra-fitting`).

.. code-block:: python

   import numpy as np
   from drdmannturb.fluctuation_generation import FluctuationFieldGenerator, MannCovariance

   covariance = MannCovariance(length_scale=50.0, E0=1.0, Gamma=3.9)

   generator = FluctuationFieldGenerator(
       grid_dimensions=np.array([1000.0, 500.0, 300.0]),  # meters
       grid_levels=np.array([6, 6, 5]),                   # 2^level + 1 points per direction
       covariance=covariance,
   )

   field = generator.generate(num_blocks=2, zref=100.0, uref=10.0, z0=0.1, windprofiletype="LOG")

Each generated block is normalized to unit RMS and superposed with a mean velocity profile
(logarithmic or power law).

Generating from a trained DRD model
-----------------------------------

``NNCovariance`` takes a trained ``SpectralTensorModel`` plus the mapping from physical wavevectors
to the model's non-dimensional domain:

.. code-block:: python

   from drdmannturb.fluctuation_generation import NNCovariance
   from drdmannturb.spectra_fitting import spectral_tensor_models as stm

   drd_model = stm.RDT_SpectralTensor.load_model("kaimal_drd_model.pt")
   scales = drd_model.calibration_scales

   covariance = NNCovariance(
       spectral_tensor_model=drd_model,
       length_scale=zref / (2 * np.pi * scales["freq_scale"]),
       amplitude=ustar**2 * scales["spectral_scale"],
   )

See Example 5 for the complete fit-then-generate workflow.

2D+3D low-frequency model
-------------------------

For marine and large-domain applications, an optional 2D low-frequency model (Syed & Mann, 2024)
can be superposed onto the 3D field by passing ``config_2d_model`` to
``FluctuationFieldGenerator``; the 2D field is generated on a much larger horizontal domain and
interpolated onto each 3D block. See Example 6.

Exporting fields
----------------

Generated fields can be saved to VTK for downstream analysis (e.g. ParaView, or conversion to CFD
inlet boundary conditions) via ``generator.save_to_vtk(path)``. The divergence of the generated
field can be checked with ``generator.evaluate_divergence(spacing)``.

Visualization helpers built on Plotly are provided in ``wind_plot``:
``plot_velocity_magnitude`` and ``plot_velocity_components``.
