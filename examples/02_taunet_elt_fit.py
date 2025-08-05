r"""
=============================
Example 2: DRD Kaimal Data Fit
=============================

In this example, we fit a neural network-based DRD model. We'll first use the Kaimal spectra to
generate synthetic data, construct the model we intend to fit to this data, and then fit the model.

We use the IEC-recommended Mann model parameters:
:math:`L/\text{zref}=0.59, \Gamma=3.9, \sigma = \alpha\epsilon^{2/3}=3.2 * (\text{zref}^{2/3} / \text{ustar}^2)`.
In this example, the exponent :math:`\nu=-\frac{1}{3}` is fixed so that :math:`\tau(\boldsymbol{k})`
matches the slope of :math:`\tau^{IEC}` for in the energy-containing range, :math:`k \rightarrow 0`.

The following example is also discussed in the `original DRD paper <https://arxiv.org/abs/2107.11046>`_.
"""  # noqa

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example. Additionally, we choose to use
# CUDA if it is available.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb.spectra_fitting import CalibrationProblem
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

#######################################################################################
# Setting Physical Parameters
# ---------------------------
# The following cell sets the necessary physical parameters
#
# :math:`L` is our characteristic length scale, :math:`\Gamma` is our characteristic
# time scale, and :math:`\sigma = \alpha\epsilon^{2/3}` is the spectrum amplitude.

zref = 90  # reference height
z0 = 0.02
zref = 90
uref = 11.4
ustar = 0.556  # friction velocity

# Scales associated with Kaimal spectrum
L = 0.593 * zref  # length scale
Gamma = 3.89  # time scale
sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # magnitude (σ = αϵ^{2/3})

print(f"Physical Parameters: {L, Gamma, sigma}")

#######################################################################################
# Data Generation
# ---------------
#
# ``DRDMannTurb`` provides a few data generation and formatting utilities. For example,
# we can generate data according to the Kaimal spectrum with the following block of code.
#
# The Kaimal spectrum defines the auto-spectra and `uw` cross-spectra components of the
# spectral tensor. By convention, the data generation functions will provide the `vw`
# and `uw` components as `NaN` values.
#
# The Kaimal spectrum is typically defined on a certain normalized domain:
# .. math::
#
#     n = \frac{k \cdot z_{\text{ref}}}{2 \pi}
#
# where :math:`z_{\text{ref}}` is the reference height. We provide several assertions
# here to show that the data generation is done correctly.
k1 = torch.logspace(-1, 2, 20, dtype=torch.float64) / zref

kaimal_data = drdmt.spectra_fitting.data_generator.generate_kaimal_spectra(
    k1=k1,
    zref=zref,
    ustar=ustar,
)

# We assert next that the frequency domain is correctly calculated.
assert kaimal_data["ops"] is not None
ops_df = kaimal_data["ops"]
assert "freq" in ops_df.columns

data_freq = ops_df["freq"].to_torch()
assert torch.allclose(
    data_freq,
    k1 * zref / (2 * torch.pi),
)

# Assert that the uu, vv, ww, and uw components are not NaN.
assert not ops_df["uu"].is_nan().any()
assert not ops_df["vv"].is_nan().any()
assert not ops_df["ww"].is_nan().any()
assert not ops_df["uw"].is_nan().any()

# Conversely, assert that the vw and uv components are NaN.
assert ops_df["vw"].is_nan().all()
assert ops_df["uv"].is_nan().all()

#######################################################################################
# Model Construction
# ------------------
#
# Models in ``DRDMannTurb`` are constructed via a ``SpectralTensorModel``
# which requires an ``EddyLifetimeModel`` and an ``EnergySpectrumModel``.
#
# In this example, we use the ``RDT_SpectralTensor`` (Rapid Distortion Theory) with
# the ``TauNet_ELT`` (neural network-based eddy lifetime function) and the
# ``VonKarman_ESM`` (standard von Karman energy spectrum).
#
# This corresponds to the architecture discussed in the `original DRD paper <https://arxiv.org/abs/2107.11046>`_.
#
# We must also provide initial values for the model's scaling parameters ``L``, ``Gamma``,
# and ``sigma``. Each of these was defined earlier.

model = stm.RDT_SpectralTensor(
    eddy_lifetime_model=stm.TauNet_ELT(
        taunet=drdmt.TauNet(n_layers=2, hidden_layer_sizes=[10, 10], activations=[nn.ReLU(), nn.ReLU()])
    ),
    energy_spectrum_model=stm.VonKarman_ESM(),
    L_init=L,
    gamma_init=Gamma,
    sigma_init=sigma,
)

#######################################################################################
# ``CalibrationProblem`` construction
# -----------------------------------
#
# We construct a ``CalibrationProblem`` object, which we use to handle the calibration
# process and specify certain numerical parameters for the code.

pb = CalibrationProblem(
    model=model,
    data=kaimal_data,
    fit_coherence=False,  # We don't have coherence data to provide here.
    loss_params=drdmt.LossParameters(
        alpha_pen2=1.0,
        beta_reg=1.0e-5,
    ),
    integration_params=drdmt.IntegrationParameters(),
    logging_directory="runs/taunet_kaimal_fit",
    output_directory="outputs",
    device="cpu",
    # device = "cuda" if torch.cuda.is_available() else "cpu"
)

##############################################################################
# Calibration
# -----------
#
# Now, to fit our model, we call the ``CalibrationProblem.calibrate`` method
# and provide several parameters for the optimization process.

pb.calibrate(
    optimizer_class=torch.optim.LBFGS,
    optimizer_kwargs={
        "line_search_fn": "strong_wolfe",
        "max_iter": 20,
        "history_size": 20,
    },
    lr=1.0,
    max_epochs=20,
    tol=1e-6,
)

##############################################################################
# Plotting
# --------
# ``DRDMannTurb`` offers built-in plotting utilities and Tensorboard integration
# which make visualizing results and various aspects of training performance
# very simple.
#
# The following will plot the fit.
# pb.plot()

# Get the original data and model predictions
original_data = pb.get_original_data()

# Generate model predictions on the original frequency domain
with torch.no_grad():
    k1_original = torch.tensor(original_data["freq"])
    k1_scaled = k1_original / pb.freq_scale
    model_prediction = pb.OPS(k1_scaled)
    model_prediction_unscaled = pb.unscale_prediction(model_prediction)

# Set up the plot with same styling as CalibrationProblem
with plt.style.context("bmh"):
    plt.rcParams.update({"font.size": 10})

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        num="Custom Spectra Fit (2x2)",
        figsize=[12, 8],
        sharex=True,
    )

    # Colors and labels matching CalibrationProblem style
    clr = ["royalblue", "crimson", "forestgreen", "mediumorchid"]
    spectra_labels = ["11", "22", "33", "13"]
    spectra_names = ["uu", "vv", "ww", "uw"]

    # Plot each component
    for i, (component, label, name, color) in enumerate(
        zip(["uu", "vv", "ww", "uw"], spectra_labels, spectra_names, clr)
    ):
        ax = axes.flatten()[i]

        # Get data and model values
        data_vals = original_data[component]
        model_vals = model_prediction_unscaled[i].cpu().detach().numpy()

        # Take absolute values for log plotting
        data_vals_abs = np.abs(data_vals)
        model_vals_abs = np.abs(model_vals)

        # Convert to numpy if needed
        if torch.is_tensor(data_vals_abs):
            data_vals_abs = data_vals_abs.cpu().detach().numpy()

        # Plot data and model
        ax.plot(
            original_data["freq"],
            data_vals_abs,
            "o",
            markersize=3,
            color=color,
            label="Data",
            alpha=0.6,
        )

        ax.plot(
            original_data["freq"],
            model_vals_abs,
            "--",
            color=color,
            label="Model",
        )

        # Set up axes properties
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(rf"$k_1 F_{{{label}}}(k_1)$")
        ax.grid(which="both")

        # Set title
        prefix = "auto-" if i < 3 else "cross-"
        ax.set_title(f"{prefix}spectra {name}")
        ax.legend()

    fig.suptitle("Custom Spectra Fit (2x2 Layout)")
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.show()

# TODO: Add a plot of the loss function
# TODO: Add model saving and loading, show here
