from pathlib import Path

import pytest
import torch

from drdmannturb import EddyLifetimeType
from drdmannturb.enums import DataType
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

fp = Path(__file__).parent


def test_kaimal_mann():
    """
    Tests data generation for Kaimal spectra under Mann parameters from the original 90s paper.
    """
    zref = 40  # reference height
    ustar = 1.773  # friction velocity

    # Scales associated with Kaimal spectrum
    L = 0.59 * zref  # length scale
    Gamma = 3.9  # time scale
    sigma = 3.2 * ustar**2.0 / zref ** (2.0 / 3.0)  # energy spectrum scale

    k1 = torch.logspace(-1, 2, 20) / zref

    Data = OnePointSpectraDataGenerator(data_points=k1, zref=zref, ustar=ustar).Data

    kaimal_mann_spectra_new = Data[1]

    kaimal_mann_spectra_true = torch.load(
        fp / "kaimal_mann_data_raw.pt", map_location=torch.device(device)
    )

    assert torch.equal(kaimal_mann_spectra_new, kaimal_mann_spectra_true)
