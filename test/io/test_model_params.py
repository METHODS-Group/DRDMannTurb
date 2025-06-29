"""Tests for io of model storage and parameter operations."""

import os
import pickle
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from drdmannturb.enums import EddyLifetimeType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting import CalibrationProblem, OnePointSpectraDataGenerator

"""
Define necessary global variables for this suite.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

L = 0.59
Gamma = 3.9
sigma = 3.4

domain = torch.logspace(-1, 2, 20)

"""
Begin tests
"""


@pytest.mark.parametrize(
    "eddylifetime",
    [EddyLifetimeType.CUSTOMMLP, EddyLifetimeType.TAUNET],
)
def test_network_paramcount(eddylifetime: EddyLifetimeType):
    """Test the number of trainable parameters in different neural network-based eddy lifetime function DRD models.

    These are static values; note that the first 3 parameters are always the length, time, and spectrum
    amplitude quantities.

    Parameters
    ----------
    eddylifetime : EddyLifetimeType
        Type of neural network used.
    """
    pb = CalibrationProblem(
        nn_params=NNParameters(hidden_layer_sizes=[10, 10]),
        prob_params=ProblemParameters(nepochs=5, eddy_lifetime=eddylifetime),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    if eddylifetime == EddyLifetimeType.CUSTOMMLP:
        assert pb.num_trainable_params() == 160

    # TODO: What about TAUNET?


@pytest.mark.slow
def test_nnparams_load_trained_TAUNET():
    """Ensures file I/O utilities are correctly reading and writing for TAUNET."""
    # Create and cd into temporary directory
    CWD_PATH = Path().cwd()
    TEMP_VAR_DIR = CWD_PATH / "TEMP_VAR_DIR"

    Path(TEMP_VAR_DIR).mkdir(parents=False, exist_ok=True)
    os.chdir(TEMP_VAR_DIR)

    def clean_exit():
        os.chdir(CWD_PATH)

        def rm_tree(pth: Path):
            for child in pth.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rm_tree(child)
            pth.rmdir()

        rm_tree(TEMP_VAR_DIR)

    # Create (A)
    pb_A = CalibrationProblem(
        nn_params=NNParameters(nlayers=2, hidden_layer_size=10, hidden_layer_sizes=[10, 10]),
        prob_params=ProblemParameters(nepochs=2, eddy_lifetime=EddyLifetimeType.TAUNET),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    k1_data_pts = domain
    Data = OnePointSpectraDataGenerator(zref=1, data_points=k1_data_pts).Data

    # Train, write out (A)
    pb_A.eval(k1_data_pts)
    pb_A.calibrate(data=Data)
    pb_A.save_model(TEMP_VAR_DIR)

    MODEL_SAVE = TEMP_VAR_DIR / "EddyLifetimeType.TAUNET_DataType.KAIMAL.pkl"

    # Ensure file exists
    assert os.path.exists(MODEL_SAVE)

    # Read in (A)
    with open("./EddyLifetimeType.TAUNET_DataType.KAIMAL.pkl", "rb") as file:
        (
            nn_params,
            prob_params,
            loss_params,
            phys_params,
            model_params,
        ) = pickle.load(file)

    pb_B = CalibrationProblem(
        nn_params=nn_params,
        prob_params=prob_params,
        loss_params=loss_params,
        phys_params=phys_params,
        device=device,
    )
    pb_B.parameters = model_params

    # Test (A) and (B) match
    assert (pb_A.parameters == pb_B.parameters).all()

    clean_exit()


@pytest.mark.slow
def test_nnparams_load_trained_CUSTOMMLP():
    """Ensures file I/O utilities are correctly reading and writing for CUSTOMMLP."""
    # Create and cd into temporary directory
    CWD_PATH = Path().cwd()
    TEMP_VAR_DIR = CWD_PATH / "TEMP_VAR_DIR"

    Path(TEMP_VAR_DIR).mkdir(parents=False, exist_ok=True)
    os.chdir(TEMP_VAR_DIR)

    def clean_exit():
        """Chdir to .. and rm -rf the temporary directory."""
        os.chdir(CWD_PATH)

        def rm_tree(pth: Path):
            """Recursive deletion for a Path to a Dir."""
            for child in pth.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rm_tree(child)
            pth.rmdir()

        rm_tree(TEMP_VAR_DIR)

    # Create (A)
    pb_A = CalibrationProblem(
        nn_params=NNParameters(nlayers=2, hidden_layer_sizes=[10, 10], activations=[nn.GELU(), nn.ReLU()]),
        prob_params=ProblemParameters(nepochs=2, eddy_lifetime=EddyLifetimeType.CUSTOMMLP),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    k1_data_pts = domain
    Data = OnePointSpectraDataGenerator(zref=1, data_points=k1_data_pts).Data

    # Train, write out (A)
    pb_A.eval(k1_data_pts)
    pb_A.calibrate(data=Data)
    pb_A.save_model(TEMP_VAR_DIR)

    MODEL_SAVE = TEMP_VAR_DIR / "EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"

    # Ensure file actually exists
    assert os.path.exists(MODEL_SAVE)

    # Read in (A)
    with open(MODEL_SAVE, "rb") as file:
        (
            nn_params,
            prob_params,
            loss_params,
            phys_params,
            model_params,
        ) = pickle.load(file)

    pb_B = CalibrationProblem(
        nn_params=nn_params,
        prob_params=prob_params,
        loss_params=loss_params,
        phys_params=phys_params,
        device=device,
    )
    pb_B.parameters = model_params

    # Test (A) and (B) match
    assert (pb_A.parameters == pb_B.parameters).all()

    clean_exit()
