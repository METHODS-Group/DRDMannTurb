# import pytest

import sys

sys.path.append("../")

from math import log

import torch.nn as nn
from configurations_util.taunet_noiseless_synthetic import (
    CONSTANTS_CONFIG,
    Gamma,
    L,
    sigma,
)

from drdmannturb.Calibration import CalibrationProblem
from drdmannturb.DataGenerator import OnePointSpectraDataGenerator


def test_network_paramcount():
    activ_list = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    config = CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [32] * 4
    config["nepochs"] = 10
    config["beta_penalty"] = 3e-2
    pb = CalibrationProblem(**config)

    assert pb.num_trainable_params() == 4228


# @pytest.mark.slow
def test_network_magnitude():
    activ_list = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    config = CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [10] * 2
    config["nepochs"] = 20
    config["beta_penalty"] = 0.0
    config["wolfe_iter"] = 10
    pb = CalibrationProblem(**config)

    parameters = pb.parameters
    parameters[:3] = [
        log(L),
        log(Gamma),
        log(sigma),
    ]  # All of these parameters are positive
    # so we can train the NN for the log of these parameters.
    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]  # np.logspace(-1, 2, 20)
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    opt_params = pb.calibrate(Data=Data, **config)

    print(pb.epoch_model_sizes)


if __name__ == "__main__":
    test_network_magnitude()
