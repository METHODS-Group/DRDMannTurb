"""Tests for io of model storage and parameter operations."""

import pickle
from pathlib import Path

import numpy as np
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
    [EddyLifetimeType.CUSTOMMLP, EddyLifetimeType.TAUNET],
)
def test_network_paramcount(eddylifetime: EddyLifetimeType):
    pb = CalibrationProblem(
        nn_params=NNParameters(hidden_layer_sizes=[10, 10]),
        prob_params=ProblemParameters(nepochs=5, eddy_lifetime=eddylifetime),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    if eddylifetime == EddyLifetimeType.CUSTOMMLP:
        assert pb.num_trainable_params() == 160
    elif eddylifetime == EddyLifetimeType.TAUNET:
        assert pb.num_trainable_params() == 27


@pytest.mark.slow
def test_nnparams_load_trained_TAUNET():
    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2, hidden_layer_size=10, hidden_layer_sizes=[10, 10]
        ),
        prob_params=ProblemParameters(nepochs=2, eddy_lifetime=EddyLifetimeType.TAUNET),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    k1_data_pts = domain
    DataPoints = [(k1, 1) for k1 in k1_data_pts]

    Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

    pb.eval(k1_data_pts)
    pb.calibrate(data=Data)

    pb.save_model(".")

    params_fp = Path().resolve() / "./EddyLifetimeType.TAUNET_DataType.KAIMAL.pkl"
    with open(params_fp, "rb") as file:
        (
            nn_params,
            prob_params,
            loss_params,
            phys_params,
            model_params,
            # _,
            # _,
        ) = pickle.load(file)

    pb_new = CalibrationProblem(
        nn_params=nn_params,
        prob_params=prob_params,
        loss_params=loss_params,
        phys_params=phys_params,
        device=device,
    )

    pb_new.parameters = model_params

    assert (pb.parameters == pb_new.parameters).all()


@pytest.mark.slow
def test_nnparams_load_trained_CUSTOMMLP():
    pb = CalibrationProblem(
        nn_params=NNParameters(
            nlayers=2, hidden_layer_sizes=[10, 10], activations=[nn.GELU(), nn.ReLU()]
        ),
        prob_params=ProblemParameters(
            nepochs=2, eddy_lifetime=EddyLifetimeType.CUSTOMMLP
        ),
        loss_params=LossParameters(alpha_pen2=1.0, alpha_pen1=1.0e-5, beta_reg=2e-4),
        phys_params=PhysicalParameters(L=L, Gamma=Gamma, sigma=sigma, domain=domain),
        device=device,
    )

    k1_data_pts = domain
    DataPoints = [(k1, 1) for k1 in k1_data_pts]

    Data = OnePointSpectraDataGenerator(data_points=DataPoints).Data

    pb.eval(k1_data_pts)
    pb.calibrate(data=Data)

    pb.save_model(".")

    params_fp = Path().resolve() / "./EddyLifetimeType.CUSTOMMLP_DataType.KAIMAL.pkl"
    with open(params_fp, "rb") as file:
        (
            nn_params,
            prob_params,
            loss_params,
            phys_params,
            model_params,
            # _,
            # _,
        ) = pickle.load(file)

    pb_new = CalibrationProblem(
        nn_params=nn_params,
        prob_params=prob_params,
        loss_params=loss_params,
        phys_params=phys_params,
        device=device,
    )

    pb_new.parameters = model_params

    assert (pb.parameters == pb_new.parameters).all()


if __name__ == "__main__":
    # test_nnparams_load_trained_TAUNET()
    test_nnparams_load_trained_CUSTOMMLP()
