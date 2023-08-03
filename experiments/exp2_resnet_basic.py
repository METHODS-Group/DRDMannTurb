import os
import sys
from math import log
from time import time

import matplotlib.pyplot as plt
import torch
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import *
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator

import constants.consts_exp2_resnet as consts_resnet

# v2: torch.set_default_device('cuda:0')
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def driver():
    start = time()

    config = consts_resnet.CONSTANTS_CONFIG

    config["hlayers"] = [1, 2]
    pb = CalibrationProblem(**config)

    parameters = pb.parameters
    parameters[:3] = [
        log(consts_resnet.L),
        log(consts_resnet.Gamma),
        log(consts_resnet.sigma),
    ]  # All of these parameters are positive
    # so we can train the NN for the log of these parameters.
    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]  # np.logspace(-1, 2, 20)
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * consts_resnet.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(
        Data=Data, **config
    )  # , OptimizerClass=torch.optim.RMSprop)

    print(f"Elapsed time : {time() - start}")

    plt.figure()

    # plt.plot( pb.loss_history_total, label="Total Loss History")
    plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE")
    plt.yscale("log")

    # NOTE: COMMENT ME OUT!
    plt.show()

    # plt.savefig(config['output_folder']+"/" + str(activ_list) + "train_history.png", format='png', dpi=100)#
    # print("+"*30)
    # print(f"Successfully finished combination {activ_list}")

    #'hlayers' : [10, 10], # ONLY NEEDED FOR CUSTOMNET OR RESNET

    #


if __name__ == "__main__":
    driver()
