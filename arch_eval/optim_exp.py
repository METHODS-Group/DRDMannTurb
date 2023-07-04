"""Conclusion: AdamW and RMSProp perform A LOT worse"""

import sys
sys.path.append('../')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',family='serif')

from pylab import *
import pickle
from math import log
import torch.nn as nn
from torch.nn import parameter

from time import time

from fracturbulence.common import *
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator

import consts

from itertools import product

from pathlib import Path

# v2: torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def driver(): 
    start = time()

    activ_list = [nn.ELU(), nn.GELU()]

    # for idx, activ_list in enumerate(list(product(consts.ACTIVATIONS, consts.ACTIVATIONS))): #[(nn.SELU(), nn.SELU())]: # zip(consts.ACTIVATIONS, consts.ACTIVATIONS)[0]: 
        # print(f"on activation function combination {idx} given by {activ_list}")

    config = consts.CONSTANTS_CONFIG
    config['activations'] = activ_list
    config['nepochs'] = 250 
    config['OptimizerClass'] = torch.optim.RMSprop
    config['lr'] = 0.1
    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [log(consts.L), log(consts.Gamma), log(consts.sigma)] #All of these parameters are positive 
    #so we can train the NN for the log of these parameters. 
    pb.parameters = parameters[:len(pb.parameters)]
    k1_data_pts = config['domain'] #np.logspace(-1, 2, 20)
    DataPoints  = [ (k1, 1) for k1 in k1_data_pts ]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    DataValues = Data[1]

    IECtau=MannEddyLifetime(k1_data_pts*consts.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)#, OptimizerClass=torch.optim.RMSprop)

    plt.figure()

        #plt.plot( pb.loss_history_total, label="Total Loss History")
    plt.plot( pb.loss_history_epochs, 'o-', label="Epochs Loss History")
    plt.legend() 
    plt.xlabel("Epoch Number")
    plt.ylabel("MSE")
    plt.yscale('log')

    plt.show() 

#        plt.savefig(config['output_folder']+"/" + str(activ_list) + "train_history.png", format='png', dpi=100)

        # print("+"*30)
        # print(f"Successfully finished combination {activ_list}")


    print(f"Elapsed time : {time() - start}")

if __name__ == '__main__':  
    from time import time  


    driver() 