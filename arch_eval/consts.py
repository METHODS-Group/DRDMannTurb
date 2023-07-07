import torch
import torch.nn as nn

torch.set_default_tensor_type('torch.cuda.FloatTensor')

from math import log
from pathlib import Path

from fracturbulence.DataGenerator import OnePointSpectraDataGenerator

savedir = Path(__file__).parent / "results"

ACTIVATIONS = [nn.ELU(), nn.ReLU(), nn.GELU(), nn.CELU(), nn.LeakyReLU(), nn.SELU()]

CONSTANTS_CONFIG = { 
    'type_EddyLifetime' :   'customMLP', # CALIBRATION : 'tauNet',  ### 'const', TwoThird', 'Mann', 'customMLP', 'tauNet'
    'type_PowerSpectra' :   'RDT', ### 'RDT', 'zetaNet', 'C3Net', 'Corrector'
    'learn_nu'          :   False, ### NOTE: Experiment 1: False, Experiment 2: True
    'plt_tau'           :   True,
    'hlayers' : [10, 10], # ONLY NEEDED FOR CUSTOMNET OR RESNET 
    'tol'               :   1.e-9, ### not important
    'lr'                :   1,     ### learning rate
    'penalty'           :   1, # CALIBRATION: 1.e-1,
    'regularization'    :   1.e-5,# CALIBRATION: 1.e-1,
    'nepochs'           :   100,
    'curves'            :   [0,1,2,3],
    'data_type'         :   'Kaimal',  # CALIBRATION: 'Custom', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    'spectra_file'      :   'Spectra.dat',
    'Uref'              :   10, # m/s
    'zref'              :   1, #m
    'domain'            :   torch.logspace(-1, 2, 20), #np.logspace(-4, 2, 40), ### NOTE: Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
    'noisy_data'        :   0.,#0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    'output_folder'     :   str(savedir), 
    'input_folder'     :   '/Users/gdeskos/work_in_progress/WindGenerator/script/'
}

zref=CONSTANTS_CONFIG['zref']; # Hub height in meters
Uref=CONSTANTS_CONFIG['Uref']; # Average Hub height velocity in m/s
Iref = 0.14
sigma1=Iref*(0.75*Uref+5.6)
Lambda1=42; # Longitudinal turbulence scale parameter at hub height

z0=0.01
ustar=0.41*Uref/log(zref/z0)

# NOTE: values taken from experiment1 in the paper 
L = 0.59 
Gamma = 3.9 
sigma = 3.2 
