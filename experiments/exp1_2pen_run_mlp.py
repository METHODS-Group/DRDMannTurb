import os
import pickle
import sys
from itertools import product
from math import log
from pathlib import Path
from time import time

import constants.consts_exp1 as consts_exp1
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from drdmannturb.Calibration import CalibrationProblem
from drdmannturb.common import *
from drdmannturb.DataGenerator import OnePointSpectraDataGenerator

# v2: torch.set_default_device('cuda:0')

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


def driver():
    start = time()

    # activ_list = [nn.GELU(), nn.GELU(), nn.GELU(), nn.GELU()]
    activ_list = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]

    config = consts_exp1.CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [32] * 4
    config["nepochs"] = 10
    config["beta_penalty"] = 3e-2
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

    #        plt.savefig(config['output_folder']+"/" + str(activ_list) + "train_history.png", format='png', dpi=100)

    # print("+"*30)
    # print(f"Successfully finished combination {activ_list}")

    print(f"Elapsed time : {time() - start}")


if __name__ == "__main__":
    from time import time

    driver()
