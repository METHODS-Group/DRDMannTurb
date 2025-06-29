"""Tests for symmetry properties of the eddy lifetime function."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from drdmannturb import EddyLifetimeType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, generate_kaimal_spectra

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


@pytest.mark.slow
def test_mann_kaimal():
    """Test the Mann-Kaimal eddy lifetime function."""
    zref = 40  # reference height
    ustar = 1.773  # friction velocity

    # Scales associated with Kaimal spectrum
    L = 0.59 * zref  # length scale
    Gamma = 3.9  # time scale
    sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # energy spectrum scale

    k1 = torch.logspace(-1, 2, 20) / zref

    pb = CalibrationProblem(
        nn_params=NNParameters(),
        prob_params=ProblemParameters(eddy_lifetime=EddyLifetimeType.MANN, nepochs=2),
        loss_params=LossParameters(),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, ustar=ustar, domain=k1),
        device=device,
    )

    Data = generate_kaimal_spectra(data_points=k1, zref=zref, ustar=ustar)

    pb.calibrate(data=Data)

    k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
    k_1 = torch.stack([k_gd, 0 * k_gd, 0 * k_gd], dim=-1)
    k_2 = torch.stack([0 * k_gd, k_gd, 0 * k_gd], dim=-1)
    k_3 = torch.stack([0 * k_gd, 0 * k_gd, k_gd], dim=-1)
    k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3 ** (1 / 2)

    pb.OPS.EddyLifetime(k_1).cpu().detach().numpy()

    tau_model2 = pb.OPS.EddyLifetime(k_2).cpu().detach().numpy()
    tau_model2_neg2 = pb.OPS.EddyLifetime(-k_2).cpu().detach().numpy()

    assert np.array_equal(tau_model2, tau_model2_neg2), "tau function is even wrt k2"

    pb.OPS.EddyLifetime(k_3).cpu().detach().numpy()
    tau_model4 = pb.OPS.EddyLifetime(k_4).cpu().detach().numpy()

    k4_n2 = torch.stack([k_gd, -k_gd, k_gd], dim=-1) / 3 ** (1 / 2)
    tau_model4_neg2 = pb.OPS.EddyLifetime(k4_n2).cpu().detach().numpy()

    assert np.array_equal(tau_model4, tau_model4_neg2), "tau function is even wrt k2"


@pytest.mark.slow
def test_synth_basic():
    """Test the synthetic data fit."""
    # Characteristic scales associated with Kaimal spectrum
    L = 0.59  # length scale
    Gamma = 3.9  # time scale
    sigma = 3.2  # energy spectrum scale

    Uref = 21.0  # reference velocity

    zref = 1  # reference height

    # We consider the range :math:`\mathcal{D} =[0.1, 100]` and sample the data points :math:`f_j \in \mathcal{D}`
    # using a logarithmic grid of :math:`20` nodes.
    domain = torch.logspace(-1, 2, 20)

    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            # Specifying the hidden layer sizes is done by passing a list of integers, as seen here.
            hidden_layer_sizes=[10, 10],
            # Specifying the activations is done similarly.
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(nepochs=10, learn_nu=False, eddy_lifetime=EddyLifetimeType.TAUNET),
        # Note that we have not activated the first order term, but this can be done by passing a value
        # for ``alpha_pen1``.
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, Uref=Uref, domain=domain),
        logging_directory="runs/synthetic_fit",
        device=device,
    )

    Data = generate_kaimal_spectra(data_points=domain, zref=zref, ustar=Uref)

    pb.calibrate(data=Data)

    k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
    k_1 = torch.stack([k_gd, 0 * k_gd, 0 * k_gd], dim=-1)
    k_2 = torch.stack([0 * k_gd, k_gd, 0 * k_gd], dim=-1)
    k_3 = torch.stack([0 * k_gd, 0 * k_gd, k_gd], dim=-1)
    k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3 ** (1 / 2)

    pb.OPS.EddyLifetime(k_1).cpu().detach().numpy()

    tau_model2 = pb.OPS.EddyLifetime(k_2).cpu().detach().numpy()
    tau_model2_neg2 = pb.OPS.EddyLifetime(-k_2).cpu().detach().numpy()

    assert np.array_equal(tau_model2, tau_model2_neg2), "tau function is even wrt k2"

    pb.OPS.EddyLifetime(k_3).cpu().detach().numpy()
    tau_model4 = pb.OPS.EddyLifetime(k_4).cpu().detach().numpy()

    k4_n2 = torch.stack([k_gd, -k_gd, k_gd], dim=-1) / 3 ** (1 / 2)
    tau_model4_neg2 = pb.OPS.EddyLifetime(k4_n2).cpu().detach().numpy()

    assert np.array_equal(tau_model4, tau_model4_neg2), "tau function is even wrt k2"


if __name__ == "__main__":
    test_synth_basic()
