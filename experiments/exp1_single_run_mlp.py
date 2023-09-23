import sys
from math import log
from time import time

import constants.consts_exp1 as consts_exp1
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from drdmannturb.Calibration import CalibrationProblem
from drdmannturb.DataGenerator import OnePointSpectraDataGenerator

sys.path.append("../")

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


def driver():
    start = time()

    activ_list = [nn.GELU(), nn.GELU(), nn.GELU()]

    config = consts_exp1.CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [32] * 4
    config["nepochs"] = 100
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

    pb.eval(k1_data_pts)

    pb.calibrate(Data=Data, **config)

    plt.figure()

    plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE")
    plt.yscale("log")

    plt.show()

    print(f"Elapsed time : {time() - start}")


if __name__ == "__main__":
    driver()
