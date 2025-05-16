
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pickle
from math import log, log10
import torch

from torch.nn import parameter

import sys
sys.path.append('./')
from source.SpectralCoherence import SpectralCoherence
from source.DataGenerator import OnePointSpectraDataGenerator, CoherenceDataGenerator
from source.Calibration import CalibrationProblem


####################################
### Configuration
####################################

config = {
    'type_EddyLifetime' :   'Mann',  ### 'const', TwoThird', 'Mann', 'tauNet'
    'type_PowerSpectra' :   'RDT', ### 'RDT', 'zetaNet', 'C3Net', 'Corrector'
    'nlayers'           :   2,
    'hidden_layer_size' :   10,
    # 'nModes'            :   5, ### number of modes in the rational function in tauNet ### deprecated
    'learn_nu'          :   False, ### NOTE: Experiment 1: False, Experiment 2: True
    'plt_tau'           :   True,
    'tol'               :   1.e-3, ### not important
    'lr'                :   1,     ### learning rate
    'penalty'           :   1.e-1,
    'regularization'    :   1.e-1,
    'nepochs'           :   200,
    'curves'            :   [0,1,2,3],
    'data_type'         :   'Kaimal', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    'domain'            :   np.logspace(-1, 2, 20), ### NOTE: Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
    'noisy_data'        :   0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    'output_folder'     :   '/Users/gdeskos/WindGenerator/data/',
    'fg_coherence'      :   True,
}
SpCoh = SpectralCoherence(**config)


####################################
#  Initialize Data
# (One-point specrtra data)
####################################

### One-point specrtra data
k1_data_pts = config['domain'] #np.logspace(-1, 2, 20)
DataPoints  = [ (k1, 1) for k1 in k1_data_pts ]
Data_OPS = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data


####################################
#  Initialize Data
# (Coherence data)
####################################

### Data points
A = 50
N_f, N_y, N_z = 50, 11, 11
k1 = torch.logspace(-3, log10(0.5), N_f, dtype=torch.float64)
# Delta_y = torch.linspace(-A, A, N_y, dtype=torch.float64)
# Delta_z = torch.linspace(-A, A, N_z, dtype=torch.float64)
Delta_y = torch.tensor([0], dtype=torch.float64)
Delta_z = torch.tensor([30], dtype=torch.float64)
DataPoints = torch.meshgrid(k1, Delta_y, Delta_z)
DataPoints = list(torch.vstack(list(map(torch.ravel, DataPoints))).T)
DataShape  = [k1.numel(), Delta_y.numel(), Delta_z.numel()]

Data_Coherence = CoherenceDataGenerator(DataPoints=DataPoints).Data
y0 = Data_Coherence[1].reshape(DataShape).flatten()

####################################
### Calibrate
####################################
pb = CalibrationProblem(**config)
opt_params = pb.calibrate(Data=Data_OPS, Data_Coherence=Data_Coherence, **config)#, OptimizerClass=torch.optim.RMSprop)



### Forward run
y1 = SpCoh(k1, Delta_y, Delta_z).cpu().detach().numpy().flatten()



plt.figure()
semilogx(k1, y0, label='Delta_y = 30, Delta_z = 0 (data)')
semilogx(k1, y1, label='Delta_y = 30, Delta_z = 0 (model)')
plt.legend()

plt.show()


exit()

zero_ind = 5
ten_ind = 6
thirty_ind = 8
fifty_ind = -1

plt.figure()
semilogx(k1, y[:,ten_ind,zero_ind], label='Delta_y = 10, Delta_z = 0')
semilogx(k1, y[:,thirty_ind,zero_ind], label='Delta_y = 30, Delta_z = 0')
semilogx(k1, y[:,fifty_ind,zero_ind], label='Delta_y = 50, Delta_z = 0')
plt.legend()


plt.figure()
semilogx(k1, y[:,zero_ind,ten_ind], label='Delta_y = 0, Delta_z = 10')
semilogx(k1, y[:,zero_ind,thirty_ind], label='Delta_y = 0, Delta_z = 30')
semilogx(k1, y[:,zero_ind,fifty_ind], label='Delta_y = 0, Delta_z = 50')
plt.legend()
plt.show()

pass

