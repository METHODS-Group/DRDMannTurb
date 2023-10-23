"""Tests for io of model storage and parameter operations."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from drdmannturb.calibration import CalibrationProblem
from drdmannturb.data_generator import OnePointSpectraDataGenerator
from drdmannturb.shared.enums import EddyLifetimeType
from drdmannturb.shared.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

L = 0.59

Gamma = 3.9
sigma = 3.4

domain = torch.logspace(-1, 2, 20)


@pytest.mark.parametrize(
    "eddylifetime",
    [EddyLifetimeType.CUSTOMMLP, EddyLifetimeType.TAUNET, EddyLifetimeType.TAURESNET],
)
def test_network_paramcount(eddylifetime: EddyLifetimeType):
    pb = CalibrationProblem(
        nn_params=NNParameters(hidden_layer_sizes=[10, 10]),
        prob_params=ProblemParameters(nepochs=5, eddy_lifetime=eddylifetime),
        loss_params=LossParameters(alpha_pen=1.0, alpha_reg=1.0e-5, beta_pen=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    if eddylifetime == EddyLifetimeType.CUSTOMMLP:
        assert pb.num_trainable_params() == 260
    elif eddylifetime == EddyLifetimeType.TAUNET:
        assert pb.num_trainable_params() == 27
    else:
        assert pb.num_trainable_params() == 4063


if __name__ == "__main__":
    for elt in [
        EddyLifetimeType.CUSTOMMLP,
        EddyLifetimeType.TAUNET,
        EddyLifetimeType.TAURESNET,
    ]:
        test_network_paramcount(elt)
