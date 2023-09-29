# %%
import sys

sys.path.append("../")
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

import pickle
from math import log
from pathlib import Path
from time import time

<<<<<<< HEAD
from pylab import *
from torch.nn import parameter

from drdmannturb.calibration import CalibrationProblem
from drdmannturb.common import *
from drdmannturb.data_generator import OnePointSpectraDataGenerator
=======
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import *
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator
from pylab import *
from torch.nn import parameter
>>>>>>> af53bc6372a2a2bdc7e4ce595385e73ece68a031

# v2: torch.set_default_device('cuda:0')
torch.set_default_tensor_type("torch.cuda.FloatTensor")
savedir = Path().resolve() / "results"

# %%
####################################
### Configuration
####################################

config = {
    "type_EddyLifetime": "tauNet",  #'TwoThird', #'tauNet', # CALIBRATION : 'tauNet',  ### 'const', TwoThird', 'Mann', 'tauNet'
    "type_PowerSpectra": "RDT",  ### 'RDT', 'zetaNet', 'C3Net', 'Corrector'
    "nlayers": 2,
    "hidden_layer_size": 10,
    # 'nModes'            :   5, ### number of modes in the rational function in tauNet ### deprecated
    "learn_nu": False,  ### NOTE: Experiment 1: False, Experiment 2: True
    "plt_tau": True,
    "tol": 1.0e-3,  ### not important
    "lr": 1,  ### learning rate
    "penalty": 1,  # CALIBRATION: 1.e-1,
    "regularization": 1.0e-5,  # CALIBRATION: 1.e-1,
    "nepochs": 50,
    "curves": [0, 1, 2, 3],
    "data_type": "Kaimal",  # CALIBRATION: 'Custom', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    "spectra_file": "Spectra.dat",
    "Uref": 10,  # m/s
    "zref": 1,  # m
    "domain": torch.logspace(
        -1, 2, 20
    ),  # np.logspace(-4, 2, 40), ### NOTE: Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
    "noisy_data": 0.0,  # 0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    "output_folder": str(savedir),
    "input_folder": "/Users/gdeskos/work_in_progress/WindGenerator/script/",
}

start = time()

pb = CalibrationProblem(**config)

# %%
####################################
### Initialize Parameters
####################################

# Calculating turbulence parameters according to IEC standards
# we assume a hub height z=150m corresponding to the IEA 15MW wind turbine hub height
zref = config["zref"]
# Hub height in meters
Uref = config["Uref"]
# Average Hub height velocity in m/s
Iref = 0.14
sigma1 = Iref * (0.75 * Uref + 5.6)
Lambda1 = 42
# Longitudinal turbulence scale parameter at hub height


# Mann model parameters
# Gamma = 3.9
# sigma = 0.55*sigma1
# L=0.8*Lambda1;


z0 = 0.01
ustar = 0.41 * Uref / log(zref / z0)

# NOTE: values taken from experiment1 in the paper
L = 0.59
Gamma = 3.9
sigma = 3.2

# NOTE: these were used in the calibration run
# L     = 14.09
# Gamma = 3.9
# sigma = 0.15174254

print(L, Gamma, sigma)

parameters = pb.parameters
parameters[:3] = [
    log(L),
    log(Gamma),
    log(sigma),
]  # All of these parameters are positive
# so we can train the NN for the log of these parameters.
pb.parameters = parameters[: len(pb.parameters)]

k1_data_pts = config["domain"]  # np.logspace(-1, 2, 20)


if config["data_type"] == "Custom":
    if config["spectra_file"] is not None:
        spectra_file = config["spectra_file"]
        print("Reading file" + spectra_file + "\n")
        CustomData = np.genfromtxt(spectra_file, skip_header=1, delimiter=",")
        f = CustomData[:, 0]
        k1_data_pts = 2 * np.pi * f / Uref

DataPoints = [(k1, 1) for k1 in k1_data_pts]
Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

### Data perturbation
data_noise_magnitude = config["noisy_data"]
if data_noise_magnitude:
    Data[1][:] *= np.exp(
        np.random.normal(loc=0, scale=data_noise_magnitude, size=Data[1].shape)
    )

DataValues = Data[1]

# %%
IECtau = MannEddyLifetime(k1_data_pts * L)
# plt.figure(1)
# plt.loglog(k1_data_pts*L,IECtau,'k')
# #plt.xlim(0.1,60)
# plt.ylabel(r'$\tau(k)$ (aribitrary units)',fontsize=18)
# plt.xlabel(r'$(kL)$',fontsize=18)
# plt.show()
# #plt.savefig('tau.png')

# %%
####################################
### Just plot
####################################
# plt.rc('text',usetex=True)
# plt.rc('font',family='serif')


kF = pb.eval(k1_data_pts)
# plt.figure(1,figsize=(10,10))
# clr=['black','blue','red']
# plt.figure(1)
# for i in range(3):
#     plt.plot(k1_data_pts, kF[i], '-', color=clr[i], label=r'$F_{0:d}$ model'.format(i+1))
#     plt.plot(k1_data_pts, DataValues[:,i,i], '--',color=clr[i],label=r'$F_{0:d}$ data'.format(i+1) )#, label=r'$F_{0:d}$ data'.format(i+1))
# plt.plot(k1_data_pts, -kF[3], '-m', label=r'-$F_{13}$ model')
# plt.plot(k1_data_pts, -DataValues[:,0,2], '--m', label=r'$-F_{13}$ data')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r'$k_1$',fontsize=22)
# #plt.xlim(0.01,2)
# #plt.ylim(0.00001,0.1)
# plt.ylabel(r'$k_1 F(k_1)/u_\ast^2$',fontsize=22)
# plt.legend(loc='lower left',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(which='both')
# plt.title("Initial Guess")
# plt.show()


# plt.savefig(config['output_folder']+'initial_guess.png',format='png',dpi=100)

# %%
opt_params = pb.calibrate(Data=Data, **config)  # , OptimizerClass=torch.optim.RMSprop)
# NOTE: THE FOLLOWING TOOK 14 MIN 44.2 SEC
# Extrapolating, the full approx 125 epochs will take me about 40 minutes
# VRAM usage is very low and volatile (consistently 600mb at 25%)

print(f"Elapsed time : {time() - start}")

# %%
plt.figure()

# plt.plot( pb.loss_history_total, label="Total Loss History")
plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
plt.legend()
plt.xlabel("Epoch Number")
plt.ylabel("MSE")
plt.yscale("log")
plt.show()


# %%
####################################
### Export
####################################
# if 'opt_params' not in locals():
#     opt_params = pb.parameters
# filename = config['output_folder'] + config['type_EddyLifetime'] + '_' + config['data_type'] + '.pkl'
# with open(filename, 'wb') as file:
#     pickle.dump([config, opt_params, Data, pb.loss_history_total, pb.loss_history_epochs], file)
