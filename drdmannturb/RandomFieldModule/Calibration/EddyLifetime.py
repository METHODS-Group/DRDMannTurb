import sys

sys.path.append(
    "/Users/bk/Work/Papers/Collaborations/2020_inletgeneration/code/source/"
)
sys.path.append("/home/khristen/Projects/Brendan/2019_inletgeneration/code/source")

from collections.abc import Callable, Iterable
from math import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from drdmannturb.RandomFieldModule.utilities.ode_solve import (
    FEM_coefficient_matrix_generator,
    Grid1D,
)

# from RandomFieldModule.Calibration.MannSpectraObjectiveFunction import MannEddyLifetime, StdEddyLifetime

"""

    Fully connected feed-forward neural network with 1 hidden layer

"""


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
        super(NeuralNet, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        if self.hidden_layer_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_layer_size).double()
            # self.actfc = nn.ReLU()
            self.actfc = nn.Tanh()
            self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size).double()
            self.fc3 = nn.Linear(hidden_layer_size, 1).double()
            self.scaling = nn.Linear(1, 1, bias=False).double()
        else:
            self.fc1 = nn.Linear(input_size, 1).double()

    def forward(self, x):
        if self.hidden_layer_size > 0:
            out = self.fc1(x)
            out = self.actfc(out)
            out = self.fc2(out)
            out = self.actfc(out)
            out = self.fc3(out)  # out == NN(k1,k2)
            out = (
                torch.norm(x, p=2, dim=-1, dtype=torch.float64).unsqueeze(-1) ** out
            )  # out = k ** NN(k1,k2)
            # out = torch.norm(x, p=2, dim=-1, dtype=torch.float64).unsqueeze(-1)**(-1) * out  # out = k^-1 * NN(k1,k2)
            out = self.scaling(out)  # out = const * k ** NN(k1,k2)
            # out = 1/out
            return out
        else:
            return 0 * self.fc1(x * 0)  # zero function
            # return self.fc1(x*0) # constant function
            # return self.fc1(x) # same as linear regression


"""

    Eddy lifetime class

    Comments:
    Used to return value and derivative information at points and frequencies

"""


class EddyLifetime:
    def __init__(self, **kwargs):
        self.tau0 = kwargs.get("tau0", lambda k: 0)
        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 16)
        self.noise_magnitude = kwargs.get("noise_magnitude", 1e-3)

        self.NN = NeuralNet(self.input_size, self.hidden_layer_size)
        self.initialize_parameters_with_noise()

    # =========================================

    @property
    def parameters(self):
        NN_parameters = parameters_to_vector(self.NN.parameters())
        with torch.no_grad():
            param_vec = NN_parameters.numpy()
        return param_vec

    @parameters.setter
    def parameters(self, param_vec):
        assert len(param_vec) > 1
        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(param_vec, dtype=torch.float64)
        vector_to_parameters(param_vec, self.NN.parameters())

    def update_parameters(self, param_vec):
        self.parameters = param_vec

    def initialize_parameters_with_noise(self):
        noise = self.noise_magnitude * np.random.randn(*self.parameters.shape)
        self.update_parameters(noise)

    # =========================================

    def __call__(self, *args):
        return self.eval(*args)

    def eval(self, *args):
        k = np.sqrt(np.sum([kj**2 for kj in args]))
        Input = self.format_input(*args)
        with torch.no_grad():
            Output = self.tau0(k) + self.NN(Input).numpy()
        return self.format_output(Output)

    def eval_deriv(self, *args):
        self.NN.zero_grad()
        Input = self.format_input(*args)
        self.NN(Input).backward()
        dtau = torch.cat([param.grad.view(-1) for param in self.NN.parameters()])
        return dtau.numpy()

    def format_input(self, *args):
        self.save_shape = args[0].shape
        Input = np.array(list(args)).reshape([len(args), -1]).T
        if len(args) == 1:
            Input = np.hstack([Input] + [np.zeros_like(Input)] * 2)
        return torch.tensor(Input, dtype=torch.float64)

    def format_output(self, out, grad=False):
        if not grad:
            return out.reshape(self.save_shape)
        else:
            return out.reshape(self.save_shape + [-1])


############################################################################
############################################################################

if __name__ == "__main__":
    pass
