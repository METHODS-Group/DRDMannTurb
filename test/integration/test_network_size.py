import sys 

sys.path.append('../')

import torch.nn as nn 

from drdmannturb.Calibration import CalibrationProblem

from configurations_util.taunet_noiseless_synthetic import CONSTANTS_CONFIG

def test_network_size(): 
    activ_list = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    config = CONSTANTS_CONFIG
    config["activations"] = activ_list
    config["hlayers"] = [32] * 4
    config["nepochs"] = 10
    config["beta_penalty"] = 3e-2
    pb = CalibrationProblem(**config)

    assert pb.num_trainable_params() == 4228 
