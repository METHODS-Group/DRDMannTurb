import os
import sys
from math import log

import arch_eval.constants.consts_exp1 as consts_exp1
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brute

from drdmannturb.Calibration import CalibrationProblem
from drdmannturb.common import MannEddyLifetime
from drdmannturb.DataGenerator import OnePointSpectraDataGenerator

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


def run_network(x: np.ndarray, *_):
    """Objective function for the scipy gridsearch

    Parameters
    ----------
    x : ndarray
        Array of value corresponding to penalty, regularization, and beta_penalty

    Returns
    -------
    float
        NN calibration loss on this
    """
    activ_list = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()]

    config = consts_exp1.CONSTANTS_CONFIG

    config["penalty"] = x[0]
    config["regularization"] = x[1]
    config["beta_penalty"] = x[2]

    config["activations"] = activ_list
    config["hlayers"] = [32] * 4

    # TODO -- CHANGE BELOW BACK TO 10
    config["nepochs"] = 1

    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [
        log(consts_exp1.L),
        log(consts_exp1.Gamma),
        log(consts_exp1.sigma),
    ]  # All of these parameters are positive
    # so we can train the NN for the log of these parameters.
    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]  # np.logspace(-1, 2, 20)
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data
    # DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * consts_exp1.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)

    # print("Reached and now returning")

    # assert pb.loss is torch.float64
    # print("pb.loss type: " + str(type(pb.loss)))
    # print("pb.loss shape: " + str(pb.loss.size()))
    # print(pb.loss)

    return pb.loss.detach().numpy()


def driver(x: np.ndarray):
    """Runs back through with `x` and plots

    Parameters
    ----------
    x : ndarray
        An ndarray array of the penalty, regularization, and beta_penalty coefficients.

    No return
    """

    activ_list = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()]

    config = consts_exp1.CONSTANTS_CONFIG

    config["activations"] = activ_list
    config["hlayers"] = [32] * 4

    config["penalty"] = x[0]
    config["regularization"] = x[1]
    config["beta_penalty"] = x[2]

    # TODO -- change the below back to 10!!!
    config["nepochs"] = 2

    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [
        log(consts_exp1.L),
        log(consts_exp1.Gamma),
        log(consts_exp1.sigma),
    ]  # All of these parameters are positive
    # so we can train the NN for the log of these parameters.
    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * consts_exp1.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(
        Data=Data, **config
    )  # , OptimizerClass=torch.optim.RMSprop)

    plt.figure()

    # plt.plot( pb.loss_history_total, label="Total Loss History")
    plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE")
    plt.yscale("log")

    plt.show()


if __name__ == "__main__":
    """Driving code."""
    print("{GRIDSEARCH} -- beginning")
    x_min = brute(run_network, (slice(0, 1), slice(0, 1), slice(0, 1.0)), None, Ns=8)

    print("{Completed}")

    # driver(x_min)
