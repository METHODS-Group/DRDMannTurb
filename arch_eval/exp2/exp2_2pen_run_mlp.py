import os
import pickle
import sys
from itertools import product
from math import log
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from pylab import *
from torch.nn import parameter

import arch_eval.consts_exp2 as consts_exp
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import *
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator

# v2: torch.set_default_device('cuda:0')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.append('../')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

def driver(): 
    start = time()

    activ_list = [nn.GELU(), nn.GELU(), nn.GELU(), nn.GELU()]

    # activ_list = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]

    config = consts_exp.CONSTANTS_CONFIG
    config['activations'] = activ_list
    config['hlayers'] = [16]*4
    config['nepochs'] = 25 
    config['beta_penalty'] = 2e-2 # don't put this to 1e-2 -- nan's out 

    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [log(consts_exp.L), log(consts_exp.Gamma), log(consts_exp.sigma)] #All of these parameters are positive 
    #so we can train the NN for the log of these parameters. 
    pb.parameters = parameters[:len(pb.parameters)]

    k1_data_pts = config['domain'] #torch.logspace(-1, 2, 20)
    spectra_file=config['spectra_file']
    print('Reading file' + spectra_file + '\n')
    CustomData=torch.tensor(np.genfromtxt(spectra_file,skip_header=1,delimiter=','))
    f=CustomData[:,0]
    k1_data_pts=2*torch.pi*f/consts_exp.Uref


    DataPoints  = [ (k1, 1) for k1 in k1_data_pts ]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    data_noise_magnitude = config['noisy_data']
    if data_noise_magnitude:
        Data[1][:] *= torch.exp(torch.tensor(np.random.normal(loc=0, scale=data_noise_magnitude, size=Data[1].shape)))


    DataValues = Data[1]

    IECtau=MannEddyLifetime(k1_data_pts*consts_exp.L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)#, OptimizerClass=torch.optim.RMSprop)

    print(f"Elapsed time : {time() - start}")

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



if __name__ == '__main__':  
    from time import time  


    driver() 