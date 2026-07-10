Spectra Fitting
===============

The ``spectra_fitting`` sub-module calibrates spectral tensor models — including neural-network-based
DRD models — to one-point spectra data (and, optionally, spectral coherence data).

Composing a model
-----------------

A spectral tensor model is composed of two parts:

- an **eddy lifetime model** (``EddyLifetimeModel``): ``TauNet_ELT`` (the neural network DRD
  lifetime), ``Mann_ELT``, ``TwoThirds_ELT``, or ``Constant_ELT``;
- an **energy spectrum model** (``EnergySpectrumModel``): ``VonKarman_ESM`` or ``Learnable_ESM``
  (learnable low- and high-:math:`k` exponents).

These are combined by ``RDT_SpectralTensor``, which applies the rapid distortion transformation and
owns the three learnable scales :math:`L`, :math:`\Gamma`, and :math:`\sigma`:

.. code-block:: python

   import drdmannturb as drdmt
   from drdmannturb.spectra_fitting import spectral_tensor_models as stm

   model = stm.RDT_SpectralTensor(
       eddy_lifetime_model=stm.TauNet_ELT(taunet=drdmt.TauNet(n_layers=2, hidden_layer_sizes=[10, 10])),
       energy_spectrum_model=stm.VonKarman_ESM(),
       L_init=50.0,
       gamma_init=3.9,
       sigma_init=0.05,
   )

Providing data
--------------

Data is provided as a dictionary of `polars <https://pola.rs>`_ DataFrames with keys ``"ops"``
(one-point spectra) and optionally ``"coherence"``. The ``ops`` frame has columns ``freq``, ``uu``,
``vv``, ``ww``, ``uw``, ``vw``, ``uv``, where the spectra are *frequency-weighted* (:math:`k_1 F_{ij}`)
and components not being fit are ``NaN``. There are three ways to obtain such a dictionary:

- ``CustomDataLoader``: load and validate your own CSV data;
- ``generate_kaimal_spectra`` / ``generate_von_karman_spectra``: generate synthetic data;
- construct the polars DataFrames directly.

``DRDMannTurb`` expects non-dimensional inputs; the mapping back to physical units is the user's
responsibility (see :doc:`field-generation` for how the calibration scales are used downstream).

Calibration
-----------

``CalibrationProblem`` wires the model and the data together and runs the optimization:

.. code-block:: python

   import torch
   from drdmannturb.spectra_fitting import CalibrationProblem

   pb = CalibrationProblem(
       model=model,
       data=kaimal_data,
       loss_params=drdmt.LossParameters(alpha_pen2=1.0, beta_reg=1e-5),
   )

   pb.calibrate(
       optimizer_class=torch.optim.LBFGS,
       optimizer_kwargs={"line_search_fn": "strong_wolfe"},
       lr=1.0,
       max_epochs=100,
   )

By default (``auto_scale=True``), the frequency domain and spectra are internally rescaled by their
medians for numerical stability; the applied factors are stored as ``pb.freq_scale`` and
``pb.spectral_scale``.

To fit spectral coherence data as well, pass ``fit_coherence=True`` together with a ``"coherence"``
DataFrame (columns ``r``, ``freq``, ``coh_u``, ``coh_v``, ``coh_w``) and a positive
``LossParameters.gamma_coherence`` weight.

Saving and loading
------------------

``pb.save_model()`` writes a ``.pt`` checkpoint containing the model weights, its architecture
configuration, and the calibration scaling factors. ``SpectralTensorModel.load_model()`` rebuilds
the model from the checkpoint:

.. code-block:: python

   path = pb.save_model()
   drd_model = stm.RDT_SpectralTensor.load_model(path)
   drd_model.calibration_scales  # {"freq_scale": ..., "spectral_scale": ...}

The reloaded model can be handed directly to ``NNCovariance`` for fluctuation field generation.

See the :doc:`API reference <../api/spectra-fitting>` for full signatures, and Examples 1–3 for
complete workflows.
