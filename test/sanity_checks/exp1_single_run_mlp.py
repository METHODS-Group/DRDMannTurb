import os
import sys
from math import log
from time import time

import arch_eval.constants.consts_exp1 as consts_exp1
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from drdmannturb.Calibration import CalibrationProblem
from drdmannturb.common import MannEddyLifetime
from drdmannturb.DataGenerator import OnePointSpectraDataGenerator

sys.path.append("../")

# plt.rc("text", usetex=True)
plt.rc("font", family="serif")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def driver():
    activ_list = [nn.GELU(), nn.GELU(), nn.GELU()]

    config = consts_exp1.CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [32] * 4
    config["nepochs"] = 0
    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [
        log(consts_exp1.L),
        log(consts_exp1.Gamma),
        log(consts_exp1.sigma),
    ]
    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]  # np.logspace(-1, 2, 20)
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data
    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * consts_exp1.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)


if __name__ == "__main__":
    driver()
