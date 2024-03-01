"""Tests for assessing data input-output accuracy. This specifically concerns the DataGenerator class."""


from pathlib import Path

import numpy as np
import pytest
import torch

from drdmannturb.enums import DataType
from drdmannturb.spectra_fitting import OnePointSpectraDataGenerator
from drdmannturb.fluctuation_generation import GenerateFluctuationField



path = Path().resolve()

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
    """
    These tests ensure that the data loading features match freshly generated
    data given the same parameters
    """

    CustomData = torch.tensor(
        np.genfromtxt(spectra_file, skip_header=1, delimiter=","), dtype=torch.float
    )
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


def test_vtk_netcdf_io_errors():
    """
    These tests ensure that the VTK file I/O routines cleanly succeed
    and fail when they should
    """
    friction_velocity = 2.683479938442173
    reference_height = 180.0
    grid_dimensions = np.array([300.0, 864.0, 576.0]) #* 1/20#* 1/10
    grid_levels = np.array([6, 6, 8])
    seed = None  # 9000
    Type_Model = "NN"
    path_to_parameters = (
        path / "../docs/source/results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
        if path.name == "examples"
        else path / "../results/EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    )

    gen_drd = GenerateFluctuationField(
        friction_velocity,
        reference_height,
        grid_dimensions,
        grid_levels,
        model=Type_Model,
        path_to_parameters=path_to_parameters,
        seed=seed
    )
    gen_drd.generate(1)

    """
    Test empty string filename fail
    """
    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_vtk("")

    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_netcdf("")

    """
    Test ill-formed file path fail
      - leads to a non-directory
    """
    import os
    f = open("DUMMY.txt", "x")
    assert os.path.exists("DUMMY.txt"), "DUMMY.txt is not where it was expected"

    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_vtk(filename="TEST", filepath="DUMMY.txt")

    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_netcdf(filename="TEST", filepath="DUMMY.txt")

    os.remove("DUMMY.txt")

    """
      - leads to a non-existent directory
    """
    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_vtk(filename="TEST", filepath=Path("./this/does/not/exist"))

    with pytest.raises(ValueError) as e_info:
        gen_drd.save_to_netcdf(filename="TEST", filepath=Path("./this/does/not/exist"))


if __name__ == "__main__":
    test_custom_spectra_load()
    test_vtk_netcdf_io_errors()
