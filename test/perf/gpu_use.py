"""
Basic tests for assessing GPU use of the package.

During the initial release, the GPU utilization during training was >=95% throughout training. Changes that drop GPU
utilization should be considered regressions to package performance.
"""

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
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

L = 0.59
Gamma = 3.9
sigma = 3.2

Uref = 21.0  # reference velocity

zref = 1  # reference height

domain = torch.logspace(-1, 2, 20)


@pytest.mark.slow
def test_gpu_utilization_synth_fit():
    """
    Test that the GPU utilization is >=95% throughout training.

    This test is a simple check to ensure that the GPU utilization is
    not too low, which would indicate that the model is not using the
    GPU effectively.
    """
    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2,
            # Specifying the hidden layer sizes is done by passing a list of integers, as seen here.
            hidden_layer_sizes=[10, 10],
            # Specifying the activations is done similarly.
            activations=[nn.ReLU(), nn.ReLU()],
        ),
        prob_params=ProblemParameters(nepochs=10, learn_nu=False, eddy_lifetime=EddyLifetimeType.TAUNET),
        # Note that we have not activated the first order term,
        # but this can be done by passing a value for ``alpha_pen1``
        loss_params=LossParameters(alpha_pen2=1.0, beta_reg=1.0e-5),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, Uref=Uref, domain=domain),
        logging_directory="runs/synthetic_fit",
        device=device,
    )

    Data = OnePointSpectraDataGenerator(zref=zref, data_points=domain).Data

    import importlib.util as util

    pb.calibrate(data=Data)

    if torch.cuda.is_available() and util.find_spec("pynvml") is not None:
        assert torch.cuda.utilization() >= 95
    else:
        raise EnvironmentError("CUDA must be available in test runner with pynvml installed in the environment.")
