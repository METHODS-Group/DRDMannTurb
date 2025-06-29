"""Tests for assessing data input-output accuracy. This specifically concerns the DataGenerator class."""

from pathlib import Path

import numpy as np
import torch

from drdmannturb.enums import DataType
from drdmannturb.spectra_fitting import OnePointSpectraDataGenerator

path = Path(__file__).parent

device = "cuda" if torch.cuda.is_available() else "cpu"

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


spectra_file = path / "../../docs/source/data/Spectra.dat"

domain = torch.logspace(-1, 3, 40)

L = 70  # length scale
GAMMA = 3.7  # time scale
SIGMA = 0.04  # energy spectrum scale

Uref = 21
zref = 1


def test_custom_spectra_load():
    """Test custom spectra loading."""
    CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","), dtype=torch.float)
    f = CustomData[:, 0]
    k1_data_pts = 2 * torch.pi * f / Uref
    Data = OnePointSpectraDataGenerator(
        zref=zref,
        data_points=k1_data_pts,
        data_type=DataType.CUSTOM,
        spectra_file=spectra_file,
        k1_data_points=k1_data_pts.data.cpu().numpy(),
    ).Data

    assert torch.equal(CustomData[:, 1], Data[1][:, 0, 0])
    assert torch.equal(CustomData[:, 2], Data[1][:, 1, 1])
    assert torch.equal(CustomData[:, 3], Data[1][:, 2, 2])
    assert torch.equal(CustomData[:, 4], -Data[1][:, 0, 2])
