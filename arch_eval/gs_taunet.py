import sys

sys.path.append('../')
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

import pickle
from itertools import product
from math import log
from pathlib import Path
from time import time

import torch.nn as nn
from pylab import *
from torch.nn import parameter

import arch_eval.consts_exp1 as consts_exp1
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import *
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator

# v2: torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def driver(): 
    for idx, activ_list in enumerate(list(product(consts_exp1.ACTIVATIONS, consts_exp1.ACTIVATIONS))): #[(nn.SELU(), nn.SELU())]: # zip(consts.ACTIVATIONS, consts.ACTIVATIONS)[0]: 
        print(f"on activation function combination {idx} given by {activ_list}")

        config = consts_exp1.CONSTANTS_CONFIG
        config['activations'] = activ_list
        pb = CalibrationProblem(**config)

        parameters = pb.parameters
        parameters[:3] = [log(consts_exp1.L), log(consts_exp1.Gamma), log(consts_exp1.sigma)] #All of these parameters are positive 
        #so we can train the NN for the log of these parameters. 
        pb.parameters = parameters[:len(pb.parameters)]

        k1_data_pts = config['domain'] #np.logspace(-1, 2, 20)
        DataPoints  = [ (k1, 1) for k1 in k1_data_pts ]
        Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

        DataValues = Data[1]

        IECtau=MannEddyLifetime(k1_data_pts*consts_exp1.L)
        kF = pb.eval(k1_data_pts)

        opt_params = pb.calibrate(Data=Data, **config)#, OptimizerClass=torch.optim.RMSprop)

        plt.figure()

        #plt.plot( pb.loss_history_total, label="Total Loss History")
        plt.plot( pb.loss_history_epochs, 'o-', label="Epochs Loss History")
        plt.legend() 
        plt.xlabel("Epoch Number")
        plt.ylabel("MSE")
        plt.yscale('log')

        plt.savefig(config['output_folder']+"/" + str(activ_list) + "train_history.png", format='png', dpi=100)

        print("+"*30)
        print(f"Successfully finished combination {activ_list}")



if __name__ == '__main__':  
    from time import time  

    start = time()

    driver() 

    print(f"Elapsed time : {time() - start}")
