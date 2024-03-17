from pathlib import Path

import numpy as np
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
from drdmannturb.spectra_fitting import OnePointSpectraDataGenerator

device = "cpu"

fp = Path(__file__).parent


def test_kaimal_mann():
    """
    Tests data generation for Kaimal spectra under Mann parameters from the original 90s paper.
    """
    zref = 40  # reference height
    ustar = 1.773  # friction velocity

    k1 = torch.logspace(-1, 2, 20) / zref

    Data = OnePointSpectraDataGenerator(data_points=k1, zref=zref, ustar=ustar).Data

    kaimal_mann_spectra_new = Data[1].to("cpu").numpy()

    kaimal_mann_spectra_true = (
        torch.load(fp / "kaimal_mann_data_raw.pt", map_location=torch.device(device))
        .to("cpu")
        .numpy()
    )

    assert np.allclose(kaimal_mann_spectra_new, kaimal_mann_spectra_true)
