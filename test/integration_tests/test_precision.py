"""Quick integration test to check that single and double precision tensors both work."""

import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb import (
    EddyLifetimeType,
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem


def test_single_precision_tensors():
    """
    Test that single precision tensors work as expected.

    Simple integration test to check that DRDMannTurb's calibration
    methods work as expected with single precision inputs.
    """
    k1 = torch.logspace(-1, 2, 20, dtype=torch.float32)

    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            hidden_layer_sizes=[10, 10],
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(
            nepochs=10,
            learn_nu=False,
            eddy_lifetime=EddyLifetimeType.TAUNET,
            num_components=4,
        ),
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(
            L=1.0,
            Gamma=1.0,
            sigma=1.0,
            ustar=1.0,
            zref=1.0,
            domain=k1,
        ),
    )

    data = drdmt.generate_kaimal_spectra(k1, 1.0, 1.0)

    assert data["k1"].dtype == torch.float32
    assert data["ops"].dtype == torch.float32

    optimal_parameters = pb.calibrate(data)

    assert optimal_parameters.dtype == torch.float32


def test_double_precision_tensors():
    """
    Test that double precision tensors work as expected.

    Simple integration test to check that DRDMannTurb's calibration
    methods work as expected with double precision inputs.
    """
    k1 = torch.logspace(-1, 2, 20, dtype=torch.float64)

    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            hidden_layer_sizes=[10, 10],
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(
            nepochs=10,
            learn_nu=False,
            eddy_lifetime=EddyLifetimeType.TAUNET,
            num_components=4,
        ),
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(
            L=1.0,
            Gamma=1.0,
            sigma=1.0,
            ustar=1.0,
            zref=1.0,
            domain=k1,
        ),
    )

    data = drdmt.generate_kaimal_spectra(k1, 1.0, 1.0)

    assert data["k1"].dtype == torch.float64
    assert data["ops"].dtype == torch.float64

    optimal_parameters = pb.calibrate(data)

    assert optimal_parameters.dtype == torch.float64
