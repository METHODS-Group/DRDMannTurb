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
from pylab import *
from scipy.special import hyp2f1
# from rational_torch import Rational
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# from drdmannturb.RandomFieldModule.utilities.ode_solve import FEM_coefficient_matrix_generator, Grid1D
# from RandomFieldModule.Calibration.MannSpectraObjectiveFunction import MannEddyLifetime, StdEddyLifetime


"""
    ==================================================================================================================
    Functions
    ==================================================================================================================
"""


@torch.jit.script
def EnergySpectrum(kL):
    nu = 17 / 6
    p = 4
    cL = 1
    # return kL**p / (1+kL**2)**nu
    return kL**p / (cL + kL**2) ** (5 / 6 + p / 2)


# ====================================================================================
# Classical Rapid Distortion Spectra
# ====================================================================================
# @torch.jit.script
def PowerSpectraRDT(k1, k2, k3, beta, E0):
    k30 = k3 + beta * k1
    # k3  = k30 - beta*k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    # arg = beta * k1 * torch.sqrt(s) / (kk0 - k30 * k1 * beta)
    # C2  = k2 * kk0 / torch.sqrt(s**3) * torch.atan(arg)
    C2 = (
        k2
        * kk0
        / torch.sqrt(s**3)
        * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)
    )

    # arg1 = k30/torch.sqrt(s)
    # arg2 = k3 /torch.sqrt(s)
    # C2  = k2 * kk0 / torch.sqrt(s**3) * (torch.atan(arg1) - torch.atan(arg2))

    zeta1 = C1 - k2 / k1 * C2
    zeta2 = C1 * k2 / k1 + C2

    Phi11 = (
        E0
        / (kk0**2)
        * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
    )
    Phi22 = (
        E0
        / (kk0**2)
        * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
    )
    Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
    Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

    Phi12 = (
        E0
        / (kk0**2)
        * (
            -k1 * k2
            - k1 * k30 * zeta2
            - k2 * k30 * zeta1
            + (k1**2 + k2**2) * zeta1 * zeta2
        )
    )
    Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

    # Phi = Phi22
    # int_Phi_dk3 = torch.trapz(Phi[i], x=k3)
    # kF = k1 * torch.trapz(int_Phi_dk3, x=k2[...,0])   ### just fix k3=0 (slices are idential in meshgrid)

    div = torch.stack(
        [
            k1 * Phi11 + k2 * Phi12 + k3 * Phi13,
            k1 * Phi12 + k2 * Phi22 + k3 * Phi23,
            k1 * Phi13 + k2 * Phi23 + k3 * Phi33,
        ]
    ) / (1 / 3 * (Phi11 + Phi22 + Phi33))

    return [Phi11, Phi22, Phi33, Phi13, Phi12, Phi23], div


# ====================================================================================
# Spectra with zeta1 and zeta2 as NeuralNets (violates div u=0 !!!)
# ====================================================================================
# @torch.jit.script
def PowerSpectraNN(k1, k2, k3, beta, E0, delta_zeta1, delta_zeta2):
    k30 = k3 + beta * k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    # arg = beta * k1 * torch.sqrt(s) / (kk0 - k30 * k1 * beta)
    # C2  = k2 * kk0 / s**(3/2) * torch.arctan(arg)

    arg1 = k30 / torch.sqrt(s)
    arg2 = k3 / torch.sqrt(s)
    C2 = k2 * kk0 / torch.sqrt(s**3) * (torch.atan(arg1) - torch.atan(arg2))

    zeta1 = C1 - k2 / k1 * C2 + delta_zeta1
    zeta2 = k2 / k1 * C1 + C2 + delta_zeta2

    # zeta1 = zeta1 * (1+delta_zeta1)
    # zeta2 = zeta2 * (1+delta_zeta2)

    Phi11 = (
        E0
        / (kk0**2)
        * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
    )
    Phi22 = (
        E0
        / (kk0**2)
        * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
    )
    Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
    Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

    Phi12 = (
        E0
        / (kk0**2)
        * (
            -k1 * k2
            - k1 * k30 * zeta2
            - k2 * k30 * zeta1
            + (k1**2 + k2**2) * zeta1 * zeta2
        )
    )
    Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

    div = torch.stack(
        [
            k1 * Phi11 + k2 * Phi12 + k3 * Phi13,
            k1 * Phi12 + k2 * Phi22 + k3 * Phi23,
            k1 * Phi13 + k2 * Phi23 + k3 * Phi33,
        ]
    ) / (1 / 3 * (Phi11 + Phi22 + Phi33))

    return [Phi11, Phi22, Phi33, Phi13], div


# ====================================================================================
# Spectra with C3 corrector
# ====================================================================================
@torch.jit.script
def PowerSpectraC3(k1, k2, k3, beta, E0, C3):
    k30 = k3 + beta * k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    # arg = beta * k1 * torch.sqrt(s) / (kk0 - k30 * k1 * beta)
    # C2  = k2 * kk0 / s**(3/2) * torch.arctan(arg)

    arg1 = k30 / torch.sqrt(s)
    arg2 = k3 / torch.sqrt(s)
    C2 = k2 * kk0 / torch.sqrt(s**3) * (torch.atan(arg1) - torch.atan(arg2))

    zeta1 = C1 - k2 / k1 * C2 - k2 / k1 * C3
    zeta2 = k2 / k1 * C1 + C2 + C3

    Phi11 = (
        E0
        / (kk0**2)
        * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
    )
    Phi22 = (
        E0
        / (kk0**2)
        * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
    )
    Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
    Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

    return Phi11, Phi22, Phi33, Phi13


# ====================================================================================
# Corrected Spectra with a general type corretor
# ====================================================================================
# @torch.jit.script
def PowerSpectraCorr(k1, k2, k3, beta, E0, Corrector):
    k30 = k3 + beta * k1
    # k3  = k30 - beta*k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    # arg = beta * k1 * torch.sqrt(s) / (kk0 - k30 * k1 * beta)
    # C2  = k2 * kk0 / torch.sqrt(s**3) * torch.atan(arg)
    C2 = (
        k2
        * kk0
        / torch.sqrt(s**3)
        * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)
    )

    # arg1 = k30/torch.sqrt(s)
    # arg2 = k3 /torch.sqrt(s)
    # C2  = k2 * kk0 / torch.sqrt(s**3) * (torch.atan(arg1) - torch.atan(arg2))

    zeta1 = C1 - k2 / k1 * C2
    zeta2 = C1 * k2 / k1 + C2
    zeta3 = kk0 / kk

    D = torch.zeros_like(Corrector)
    D[..., 0, 0] = 1
    D[..., 1, 1] = 1
    D[..., 0, 2] = zeta1
    D[..., 1, 2] = zeta2
    D[..., 2, 2] = zeta3

    P = torch.zeros_like(Corrector)
    P[..., 0, 0] = 1 - k1 * k1 / kk0
    P[..., 1, 1] = 1 - k2 * k2 / kk0
    P[..., 2, 2] = 1 - k30 * k30 / kk0
    P[..., 1, 0] = -k1 * k2 / kk0
    P[..., 2, 0] = -k1 * k30 / kk0
    P[..., 0, 1] = P[..., 1, 0]
    P[..., 2, 1] = -k2 * k30 / kk0
    P[..., 0, 2] = P[..., 2, 0]
    P[..., 1, 2] = P[..., 2, 1]

    tD = D + Corrector

    scale = E0 / kk0
    sqrtPhi = torch.matmul(tD, P)
    Phi = torch.matmul(sqrtPhi, sqrtPhi.transpose(-2, -1)) * scale[..., None, None]

    # Phi11 = E0/(kk0**2) * (kk0 - k1**2 - 2*k1*k30*zeta1 + (k1**2+k2**2)*zeta1**2)
    # Phi22 = E0/(kk0**2) * (kk0 - k2**2 - 2*k2*k30*zeta2 + (k1**2+k2**2)*zeta2**2)
    # Phi33 = E0/(kk**2)  * (k1**2 + k2**2)
    # Phi13 = E0/(kk*kk0)  * (-k1*k30 + (k1**2+k2**2)*zeta1)

    div = 0

    # return Phi[...,0,0], Phi[...,1,1], Phi[...,2,2], Phi[...,0,2], Phi[...,0,1], Phi[...,1,2]
    return [
        Phi[..., 0, 0],
        Phi[..., 1, 1],
        Phi[..., 2, 2],
        Phi[..., 0, 2],
        Phi[..., 0, 1],
        Phi[..., 1, 2],
    ], div


# ====================================================================================
# Mann's Eddy Liftime
# ====================================================================================


def MannEddyLifetime(kL):
    return (kL) ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -((kL) ** (-2))))


# LifeTime[kL_] := Module[{kLloc = Max[0.005, kL], kSqr},
#   kSqr = kLloc^2;
#   (1 + kSqr)^(1/6)/
#     kLloc*(1.2050983316598936 - 0.04079766636961979*kLloc +
#       1.1050803451576134*kSqr)/(1 - 0.04103886513006046*kLloc +
#       1.1050902034670118*kSqr)]


"""
    ==================================================================================================================
    Fully connected feed-forward neural networks
    ==================================================================================================================
"""


class Rational(nn.Module):
    def __init__(self, nModes=20):
        super().__init__()
        self.nModes = nModes
        self.poles = torch.linspace(0, 100, self.nModes, dtype=torch.float64)
        self.poles = nn.Parameter(self.poles)
        self.weights = nn.Parameter(torch.zeros((self.nModes,), dtype=torch.float64))

    def forward(self, x):
        den = x.abs().unsqueeze(-1) + self.poles**2
        out = self.weights / den
        out = out.sum(dim=-1)
        return out


# ========================================================
# NN for Eddy Lifetime
# ========================================================


class tauNet(nn.Module):
    def __init__(self, **kwargs):
        super(tauNet, self).__init__()

        self.hidden_layer_size = kwargs.get("hidden_layer_size", 10)
        self.nModes = kwargs.get("nModes", 10)
        self.degree = kwargs.get("degree", 3)

        # self.actfc = torch.cos
        self.actfc = nn.ReLU()
        # self.actfc = nn.Softplus()
        # self.actfc = nn.Tanh()
        # self.actfc = Rational()
        # self.actfc = nn.Sigmoid()
        self.Ractfc = Rational(nModes=self.nModes)
        # self.Ractfc= nn.Sigmoid()
        self.fc0 = nn.Linear(
            3 * self.degree, self.hidden_layer_size, bias=False
        ).double()
        self.fc1 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc2 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc3 = nn.Linear(self.hidden_layer_size, 1, bias=False).double()

        # self.fc0   = nn.Bilinear(3*self.degree, 3*self.degree, self.hidden_layer_size, bias=False).double()
        # self.fc1   = nn.Bilinear(self.hidden_layer_size, self.hidden_layer_size, self.hidden_layer_size, bias=False).double()
        # self.fc2   = nn.Bilinear(self.hidden_layer_size, self.hidden_layer_size, self.hidden_layer_size, bias=False).double()
        # self.fc3   = nn.Bilinear(self.hidden_layer_size, self.hidden_layer_size, 1, bias=False).double()

        self.power = 1 + torch.arange(self.degree, dtype=torch.float64)

    def forward(self, k, kL):
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k1**2 + k2**2 + k3**2

        x = (k.unsqueeze(-1) ** self.power).flatten(start_dim=-2, end_dim=-1)
        out = self.fc0(x)  # ,x)
        out = self.actfc(out)
        out = self.fc1(out)  # ,out)
        out = self.actfc(out)
        out = self.fc2(out)  # ,out)
        out = self.actfc(out)
        out = self.fc3(out)  # ,out)
        out = self.actfc(out)
        out = self.Ractfc(out)

        # return out*torch.exp(-kk).unsqueeze(-1) + kk.unsqueeze(-1)**(-1/3) #+ (1-out)*kk.unsqueeze(-1)**(-1/2)
        # return out * kk.unsqueeze(-1)**(-1/3) + (1-out)*kk.unsqueeze(-1)**(-1/2)

        base = torch.tensor(
            MannEddyLifetime(kL.detach().numpy()), dtype=torch.float64
        )  # .unsqueeze(-1)
        # base = kk.unsqueeze(-1)**(-1/3)
        return out + base


# ========================================================
# NN for the Energy Spectrum
# ========================================================


class energyNet(nn.Module):
    def __init__(self, **kwargs):
        super(energyNet, self).__init__()

        self.hidden_layer_size = kwargs.get("hidden_layer_size", 10)
        self.nModes = kwargs.get("nModes", 10)

        self.actfc = nn.ReLU()
        self.Ractfc = Rational(nModes=self.nModes)
        self.fc0 = nn.Linear(
            3 * self.degree, self.hidden_layer_size, bias=False
        ).double()
        self.fc1 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc2 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc3 = nn.Linear(self.hidden_layer_size, 1, bias=False).double()

    def forward(self, k, kL):
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k1**2 + k2**2 + k3**2

        x = (k.unsqueeze(-1) ** self.power).flatten(start_dim=-2, end_dim=-1)
        out = self.fc0(x)  # ,x)
        out = self.actfc(out)
        out = self.fc1(out)  # ,out)
        out = self.actfc(out)
        out = self.fc2(out)  # ,out)
        out = self.actfc(out)
        out = self.fc3(out)  # ,out)
        out = self.actfc(out)
        out = self.Ractfc(out)

        # return out*torch.exp(-kk).unsqueeze(-1) + kk.unsqueeze(-1)**(-1/3) #+ (1-out)*kk.unsqueeze(-1)**(-1/2)
        # return out * kk.unsqueeze(-1)**(-1/3) + (1-out)*kk.unsqueeze(-1)**(-1/2)

        base = torch.tensor(
            MannEddyLifetime(kL.detach().numpy()), dtype=torch.float64
        )  # .unsqueeze(-1)
        # base = kk.unsqueeze(-1)**(-1/3)
        return out + base


# ========================================================
# NN for correctors of zeta_1 and zeta_2
# ========================================================


class zetaNet(nn.Module):
    def __init__(self, **kwargs):
        super(zetaNet, self).__init__()

        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 0)

        self.actfc = nn.ReLU()
        # self.fc0   = nn.Linear(3, 1).double()
        self.fc0 = nn.Linear(3, self.hidden_layer_size).double()
        self.fc1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size).double()
        self.fc2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size).double()
        self.fc3 = nn.Linear(self.hidden_layer_size, 1).double()

        # self.nModes = 20
        # self.fc_d   = nn.Linear(1, self.nModes, bias=False).double()
        # self.fc_c   = nn.Linear(1, self.nModes, bias=False).double()

    def forward(self, k):
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k1**2 + k2**2 + k3**2
        out = self.fc0(k)
        out = self.actfc(out)
        out = self.fc1(out)
        out = self.actfc(out)
        out = self.fc2(out)
        out = self.actfc(out)
        out = self.fc3(out)
        return out.squeeze(-1) / kk
        # ones= torch.ones_like(out)
        # d   = self.fc_d(ones)
        # c   = self.fc_c(ones)
        # z   = out.abs().squeeze(-1)
        # out = c**2 / (z[...,None] + d.abs())
        # out = out.sum(dim=-1)
        # return out


# ========================================================
# NN for C3 corrector
# ========================================================


class NET_C3_Corrector(nn.Module):
    def __init__(self, **kwargs):
        super(NET_C3_Corrector, self).__init__()

        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 0)

        self.actfc = nn.ReLU()
        self.fc0 = nn.Linear(3, self.hidden_layer_size).double()
        self.fc1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size).double()
        self.fc2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size).double()
        self.fc3 = nn.Linear(self.hidden_layer_size, 1).double()

        # self.nModes = 40
        # self.fc_d   = nn.Linear(1, self.nModes, bias=False).double()
        # self.fc_c   = nn.Linear(self.nModes, 1).double()

    def forward(self, k):
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k1**2 + k2**2 + k3**2
        out = self.fc0(k)
        out = self.actfc(out)
        out = self.fc1(out)
        out = self.actfc(out)
        out = self.fc2(out)
        out = self.actfc(out)
        out = self.fc3(out)
        return (0.1 + out.squeeze(-1)) / (1 + kk)
        # ones = torch.ones_like(kk).unsqueeze(-1)
        # d   = self.fc_d(ones)
        # out = self.fc_c(1 / (kk[...,None] + d**2))
        # return (0.01 + out.squeeze(-1))


# ========================================================
# NN for the full spectra corrector
# ========================================================


class NET_Corrector(nn.Module):
    def __init__(self, **kwargs):
        super(NET_Corrector, self).__init__()
        self.eps = nn.ModuleList([NET_Simple(**kwargs) for i in range(6)])
        # self.eps = NET_Simple(**kwargs)

    def forward(self, k):
        k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]
        kk = k1**2 + k2**2 + k3**2
        # sqrt_kk = kk #torch.sqrt(kk)
        # k1, k2, k3 = k1/sqrt_kk, k2/sqrt_kk, k3/sqrt_kk
        eps21 = self.eps[0](k)
        eps31 = self.eps[1](k)
        eps12 = self.eps[2](k)
        eps32 = self.eps[3](k)
        eps13 = self.eps[4](k)
        eps23 = self.eps[5](k)
        # --------------------------
        # eps = self.eps(k)
        # eps21 = eps[...,0]
        # eps31 = eps[...,1]
        # eps12 = eps[...,2]
        # eps32 = eps[...,3]
        # eps13 = eps[...,4]
        # eps23 = eps[...,5]
        # --------------------------
        Eps11 = k2 * eps21 + k3 * eps31
        Eps21 = -k1 * eps21
        Eps31 = -k1 * eps31
        Eps12 = -k2 * eps12
        Eps22 = k1 * eps12 + k3 * eps32
        Eps32 = -k2 * eps32
        Eps13 = -k3 * eps13
        Eps23 = -k3 * eps23
        Eps33 = k1 * eps13 + k2 * eps23
        # --------------------------
        # eps21 = self.eps[0](k)
        # eps31 = self.eps[1](k)
        # eps22 = self.eps[2](k)
        # eps32 = self.eps[3](k)
        # eps23 = self.eps[4](k)
        # eps33 = self.eps[5](k)
        # Eps11 = k2/k1*eps21 + k3/k1*eps31
        # Eps21 = -1*eps21
        # Eps31 = -1*eps31
        # Eps12 = k2/k1*eps22 + k3/k1*eps32
        # Eps22 = -1*eps22
        # Eps32 = -1*eps32
        # Eps13 = k2/k1*eps23 + k3/k1*eps33
        # Eps23 = -1*eps23
        # Eps33 = -1*eps33
        # --------------------------
        col1 = torch.stack([Eps11, Eps21, Eps31], dim=-1)
        col2 = torch.stack([Eps12, Eps22, Eps32], dim=-1)
        col3 = torch.stack([Eps13, Eps23, Eps33], dim=-1)
        out = torch.stack([col1, col2, col3], dim=-1)
        return out / kk[..., None, None]


class NET_Simple(nn.Module):
    def __init__(self, **kwargs):
        super(NET_Simple, self).__init__()

        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 10)
        self.nModes = kwargs.get("nModes", 10)
        self.degree = kwargs.get("degree", 3)

        self.actfc = nn.Tanh()  # nn.Softplus()
        # self.Ractfc= Rational(nModes=self.nModes)
        # self.Ractfc= nn.Tanh()
        self.fc0 = nn.Linear(
            3 * self.degree, self.hidden_layer_size, bias=False
        ).double()
        self.fc1 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc2 = nn.Linear(
            self.hidden_layer_size, self.hidden_layer_size, bias=False
        ).double()
        self.fc3 = nn.Linear(self.hidden_layer_size, 1, bias=False).double()

        self.power = 1 + torch.arange(self.degree, dtype=torch.float64)

    def forward(self, k):
        kk = k.norm(-1)
        x = (k.unsqueeze(-1) ** self.power).flatten(start_dim=-2, end_dim=-1)
        out = self.fc0(x)
        out = self.actfc(out)
        out = self.fc1(out)
        out = self.actfc(out)
        out = self.fc2(out)
        out = self.actfc(out)
        out = self.fc3(out)
        out = self.actfc(out)
        out = out.squeeze(-1) * torch.exp(-kk)
        return out


# ==============================================================================================
# Neural Net for One-Point Spectra
# ==============================================================================================


class NET_OnePointSpectra(nn.Module):
    def __init__(self, **kwargs):
        super(NET_OnePointSpectra, self).__init__()

        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 0)
        self.case_EddyLifetime = kwargs.get("case_EddyLifetime", "TwoThird")
        self.case_PowerSpectra = kwargs.get("case_PowerSpectra", "RDT")

        self.init_grids()
        self.init_nets()

        if self.case_EddyLifetime == "tauNet":
            self.tauNet = tauNet(**kwargs)
        if self.case_PowerSpectra == "zetaNet":
            self.zetaNet1 = zetaNet(**kwargs)
            self.zetaNet2 = zetaNet(**kwargs)
        if self.case_PowerSpectra == "C3Net":
            self.C3Net = NET_C3_Corrector(**kwargs)
        if self.case_PowerSpectra == "Corrector":
            self.Corrector = NET_Corrector(**kwargs)

    def init_grids(self):
        p1, p2, N = -8, 0, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64) * 1.0e4
        # grid_plus = torch.linspace(1, 1000, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus))
        # self.grid_k2 = torch.cat((grid_minus, grid_plus))
        # self.grid_k2 = torch.cat((grid_zero, grid_plus))
        # self.grid_k2 = torch.cat((grid_minus[-2:], grid_zero, grid_plus[:2]))
        # self.grid_k2 = torch.cat((grid_zero, grid_plus))
        # self.grid_k2 = torch.cat((grid_minus, grid_plus))

        p1, p2, N = -8, 0, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64) * 1.0e4
        # grid_plus = torch.linspace(1, 1000, N, dtype=torch.float64)
        grid_minus = -torch.flip(grid_plus, dims=[0])
        self.grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus))
        # self.grid_k3 = torch.cat((grid_minus, grid_plus))
        # self.grid_k3 = torch.cat((grid_zero, grid_plus))

        # self.grid_k2 = torch.tensor([0.001], dtype=torch.float64)
        # self.grid_k3 = torch.tensor([-0.001], dtype=torch.float64)

        # self.grid_k2 = torch.tensor([0.03], dtype=torch.float64)
        # self.grid_k3 = torch.tensor([0.03], dtype=torch.float64)

        self.meshgrid23 = torch.meshgrid(self.grid_k2, self.grid_k3)

    def init_nets(self):
        self.Theta = torch.tensor(1, dtype=torch.float64)
        self.Scale = torch.tensor(1, dtype=torch.float64)
        self.Gamma = torch.tensor(1, dtype=torch.float64)

        self.Theta = nn.Parameter(self.Theta)
        self.Scale = nn.Parameter(self.Scale)
        self.Gamma = nn.Parameter(self.Gamma)
        # if self.case_EddyLifetime != 'tauNet': self.Gamma = nn.Parameter(self.Gamma)

    @torch.jit.export
    def EddyLifetime(self, k):
        if self.case_EddyLifetime == "const":
            kL = self.Theta.abs() * torch.norm(k, dim=-1, keepdim=True)
            tau = torch.ones_like(kL)
        elif (
            self.case_EddyLifetime == "Mann"
        ):  ### uses numpy - can not be backpropagated !!
            kL = self.Theta.abs() * torch.norm(k, dim=-1, keepdim=True)
            tau = torch.tensor(
                MannEddyLifetime(kL.detach().numpy()), dtype=torch.float64
            )
        elif self.case_EddyLifetime == "TwoThird":
            kL = self.Theta.abs() * torch.norm(k, dim=-1, keepdim=True)
            tau = kL ** (-2 / 3)
        elif self.case_EddyLifetime == "tauNet":
            kL = self.Theta.abs() * torch.norm(k, dim=-1, keepdim=True)
            tau = self.tauNet(k, kL)
        else:
            raise Exception("Wrong EddyLifetime model!")
        beta = self.Gamma.abs() * tau
        return beta.squeeze(dim=-1)

    def forward(self, k1_input):
        k1 = k1_input

        k = torch.stack(torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1)
        k0 = k.clone().detach()
        beta = self.EddyLifetime(k)
        k0[..., 2] = k[..., 2] + beta * k[..., 0]
        # beta = self.EddyLifetime(k0)
        # k[...,2] = k0[...,2] - beta*k0[...,0]
        # k0L = self.apply_Theta(torch.norm(k0, dim=-1, keepdim=True))
        k0L = self.Theta.abs() * torch.norm(k0, dim=-1, keepdim=True)
        # E0  = self.apply_Scale(EnergySpectrum(k0L)).squeeze(dim=-1)
        E0 = (self.Scale.abs() * EnergySpectrum(k0L)).squeeze(dim=-1)

        # k0 = torch.stack(torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1)
        # k0L= self.apply_Theta(torch.norm(k0, dim=-1, keepdim=True))
        # E0 = self.apply_Scale(EnergySpectrum(k0L)).squeeze(dim=-1)
        # beta = self.EddyLifetime(k0)
        # k = k0.clone().detach()
        # k[...,2] = k0[...,2] - beta*k0[...,0]

        if self.case_PowerSpectra == "RDT":
            Phi, self.div = PowerSpectraRDT(k[..., 0], k[..., 1], k[..., 2], beta, E0)
        elif self.case_PowerSpectra == "zetaNet":
            delta_zeta1 = self.zetaNet1(k)
            delta_zeta2 = self.zetaNet2(k)
            Phi, self.div = PowerSpectraNN(
                k[..., 0], k[..., 1], k[..., 2], beta, E0, delta_zeta1, delta_zeta2
            )
        elif self.case_PowerSpectra == "C3Net":
            C3 = self.C3Net(k)
            Phi, self.div = PowerSpectraC3(
                k[..., 0], k[..., 1], k[..., 2], beta, E0, C3
            )
        elif self.case_PowerSpectra == "Corrector":
            Corrector = self.Corrector(k)
            # Corrector = 0*Corrector
            Phi, self.div = PowerSpectraCorr(
                k[..., 0], k[..., 1], k[..., 2], beta, E0, Corrector
            )

        ### Integration in k2 and k3
        kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
        for i in range(len(Phi)):
            # int_Phi_dk3 = torch.trapz(Phi[i], x=k[...,1], dim=-2)
            # kF[i] = k1 * torch.trapz(int_Phi_dk3, x=k0[:,0,:,2], dim=-1)   ### just fix k3=0 (slices are idential in meshgrid)
            int_Phi_dk3 = torch.trapz(Phi[i], x=k[..., 2], dim=-1)
            kF[i] = k1 * torch.trapz(
                int_Phi_dk3, x=k[..., 0, 1], dim=-1
            )  ### just fix k3=0 (slices are idential in meshgrid)

        ### Integration in k2 and k3
        # kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
        # for i in range(len(Phi)):
        #     int_Phi_dk3 = (0.5*(Phi[i][...,1:]+Phi[i][...,:-1])*(k[...,1:,2]-k[...,:-1,2])).sum(-1)
        #     kF[i] = k1 * (0.5*(int_Phi_dk3[...,1:]+int_Phi_dk3[...,:-1])*(k[...,1:,0,1]-k[...,:-1,0,1])).sum(-1)   ### just fix k3=0 (slices are idential in meshgrid)

        ### Integration in k2 and k3
        # kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
        # mid_k3 = 0.5*(k[...,1:,2]+k[...,:-1,2])
        # dk3 = mid_k3[...,1:]-mid_k3[...,:-1]
        # mid_k2 = 0.5*(k[...,1:,0,1]+k[...,:-1,0,1])
        # dk2 = mid_k2[...,1:]-mid_k2[...,:-1]
        # for i in range(len(Phi)):
        #     int_Phi_dk3 = (Phi[i][...,1:-1]*dk3).sum(-1)
        #     kF[i] = k1 * (int_Phi_dk3[...,1:-1]*dk2).sum(-1)   ### just fix k3=0 (slices are idential in meshgrid)

        return kF

    # def ReynoldsStress(self, k1_input): ### not ready!!!
    #     k1_input = self.grid_k2
    #     k1 = k1_input

    #     ls_t = torch.linspace(0,0.01,5)
    #     I = torch.zeros(4, len(ls_t), dtype=torch.float64)

    #     for ib, t in enumerate(ls_t):
    #         k = torch.stack(torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1)
    #         beta = t*self.EddyLifetime(k)
    #         k0 = k.clone().detach()
    #         k0[...,2] = k[...,2] + beta*k[...,0]
    #         k0L = self.apply_Theta(torch.norm(k0, dim=-1, keepdim=True))
    #         E0  = self.apply_Scale(EnergySpectrum(k0L)).squeeze(dim=-1)

    #         # k  = torch.stack(torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1)
    #         # kL = self.apply_Theta(torch.norm(k, dim=-1, keepdim=True))
    #         # E0 = self.apply_Scale(EnergySpectrum(kL)).squeeze(dim=-1)
    #         # beta = self.EddyLifetime(k)

    #         if self.case_PowerSpectra == 'RDT':
    #             Phi, self.div = PowerSpectraRDT(k[...,0], k[...,1], k[...,2], beta, E0)
    #         elif self.case_PowerSpectra == 'zetaNet':
    #             delta_zeta1 = self.zetaNet1(k)
    #             delta_zeta2 = self.zetaNet2(k)
    #             Phi, self.div = PowerSpectraNN(k[...,0], k[...,1], k[...,2], beta, E0, delta_zeta1, delta_zeta2)
    #         elif self.case_PowerSpectra == 'C3Net':
    #             C3  = self.C3Net(k)
    #             Phi = PowerSpectraC3(k[...,0], k[...,1], k[...,2], beta, E0, C3)
    #         elif self.case_PowerSpectra == 'Corrector':
    #             Corrector = self.Corrector(k)
    #             # Corrector = 0*Corrector
    #             Phi = PowerSpectraCorr(k[...,0], k[...,1], k[...,2], beta, E0, Corrector)

    #         ### Integration in k2 and k3
    #         kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
    #         for i in range(len(Phi)):
    #             int_Phi_dk3 = torch.trapz(Phi[i], x=k[...,2])
    #             # kF[i] = k1 * torch.trapz(int_Phi_dk3, x=k[...,0,1])   ### just fix k3=0 (slices are idential in meshgrid)
    #             kF[i] = torch.trapz(int_Phi_dk3, x=k[...,0,1])   ### just fix k3=0 (slices are idential in meshgrid)
    #             I[i,ib] = torch.trapz(kF[i], x=k[:,0,0,0])   ### just fix k3=0 (slices are idential in meshgrid)

    #     ### Integration in k2 and k3
    #     # kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
    #     # for i in range(len(Phi)):
    #     #     int_Phi_dk3 = (0.5*(Phi[i][...,1:]+Phi[i][...,:-1])*(k[...,1:,2]-k[...,:-1,2])).sum(-1)
    #     #     kF[i] = k1 * (0.5*(int_Phi_dk3[...,1:]+int_Phi_dk3[...,:-1])*(k[...,1:,0,1]-k[...,:-1,0,1])).sum(-1)   ### just fix k3=0 (slices are idential in meshgrid)

    #     ### Integration in k2 and k3
    #     # kF = torch.zeros(len(Phi), len(k1_input), dtype=torch.float64)
    #     # mid_k3 = 0.5*(k[...,1:,2]+k[...,:-1,2])
    #     # dk3 = mid_k3[...,1:]-mid_k3[...,:-1]
    #     # mid_k2 = 0.5*(k[...,1:,0,1]+k[...,:-1,0,1])
    #     # dk2 = mid_k2[...,1:]-mid_k2[...,:-1]
    #     # for i in range(len(Phi)):
    #     #     int_Phi_dk3 = (Phi[i][...,1:-1]*dk3).sum(-1)
    #     #     kF[i] = k1 * (int_Phi_dk3[...,1:-1]*dk2).sum(-1)   ### just fix k3=0 (slices are idential in meshgrid)

    #     # return kF
    #     return I


"""
    ==================================================================================================================
    One-point spectrum class

    Comments:
    Used to return value and derivative information at points and frequencies
    ==================================================================================================================

"""

# ==============================================================================================
# Loss funtion for calibration
# ==============================================================================================


class myLossFunc:
    def __init__(self, **kwargs):
        pass

    def __call__(self, output, target, pen=None):
        loss = torch.sum((torch.log(output) - torch.log(target)) ** 2)
        # loss = torch.sum( (output-target)**2 )
        # loss = torch.sum( (output-target)**2 / torch.clamp(target**2, min=1.e-8) )
        # loss = torch.sum( ((output-target)/target)**2 )
        if pen is not None:
            loss = loss + pen
        return loss


# ==============================================================================================
# 1-point spectra class itself
# ==============================================================================================


class OnePointSpectra:
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input_size", 3)
        self.hidden_layer_size = kwargs.get("hidden_layer_size", 0)
        self.init_with_noise = kwargs.get("init_with_noise", False)
        self.noise_magnitude = kwargs.get("noise_magnitude", 1.0e-3)

        self.NN = NET_OnePointSpectra(**kwargs)
        self.init_device()
        if self.init_with_noise:
            self.initialize_parameters_with_noise()

        self.vdim = 3

    def init_device(self):
        # enable gpu device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.NN.to(device)

    # =========================================

    @property
    def parameters(self):
        NN_parameters = parameters_to_vector(self.NN.parameters())
        with torch.no_grad():
            param_vec = NN_parameters.cpu().numpy()
        return param_vec

    @parameters.setter
    def parameters(self, param_vec):
        assert len(param_vec) >= 1
        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(param_vec, dtype=torch.float64)
        vector_to_parameters(param_vec, self.NN.parameters())

    def update_parameters(self, param_vec):
        self.parameters = param_vec

    def initialize_parameters_with_noise(self):
        noise = self.noise_magnitude * np.random.randn(*self.parameters.shape)
        noise = torch.tensor(noise, dtype=torch.float64)
        # self.update_parameters(noise**2)
        vector_to_parameters(noise.abs(), self.NN.parameters())
        # try:
        #     noise2 = 100*noise.clone().detach()
        #     noise2[:] = 0.5
        #     vector_to_parameters(noise2, self.NN.zetaNet1.parameters())
        #     vector_to_parameters(noise2, self.NN.zetaNet2.parameters())
        # except: pass
        try:
            # noise[:] = 0.01
            vector_to_parameters(noise, self.NN.tauNet.parameters())
        except:
            pass
        try:
            # noise[:] = 0.1
            vector_to_parameters(noise.abs(), self.NN.Corrector.parameters())
        except:
            pass

    # =========================================

    def __call__(self, k1):
        return self.eval(k1)

    def eval(self, k1):
        Input = self.format_input(k1)
        with torch.no_grad():
            Output = self.NN(Input)
        return self.format_output(Output)

    def eval_grad(self, k1):
        self.NN.zero_grad()
        Input = self.format_input(k1)
        self.NN(Input).backward()
        grad = torch.cat([param.grad.view(-1) for param in self.NN.parameters()])
        return self.format_output(grad)

    def format_input(self, k1):
        if np.isscalar(k1):
            return torch.tensor([k1], dtype=torch.float64)
        else:
            return torch.tensor(k1, dtype=torch.float64)

    def format_output(self, out):
        return out.numpy()

    ###-----------------------------------------
    ### Clibration method
    ###-----------------------------------------
    def Calibrate(self, **kwargs):
        print("\nCallibrating MannNet...")

        DataPoints, DataValues = kwargs.get("Data")
        OptimizerClass = kwargs.get("OptimizerClass", torch.optim.LBFGS)
        lr = kwargs.get("lr", 1e-1)
        tol = kwargs.get("tol", 1e-3)
        show = kwargs.get("show", False)

        self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[:, 0].squeeze()
        self.kF_data_vals = torch.tensor(
            [DataValues[:, i, i] for i in range(3)] + [DataValues[:, 0, 2]],
            dtype=torch.float64,
        )

        k1_data_pts, y_data0 = self.k1_data_pts, self.kF_data_vals

        y = self.NN(k1_data_pts)
        y_data = torch.zeros_like(y)
        y_data[:4, ...] = y_data0

        # self.loss_fn = torch.nn.MSELoss(reduction='sum')
        # self.loss_fn = torch.nn.NLLLoss(reduction='sum')

        self.loss_fn = myLossFunc()

        if False:  # self.NN.case_PowerSpectra == 'zetaNet':
            for i in (2, 1, 0):
                if i == 2:
                    print("\n=================================")
                    print("Calibration F3.")
                    print("=================================\n")
                    optimizer = OptimizerClass(self.NN.parameters(), lr=lr)
                elif i == 1:
                    print("\n=================================")
                    print("Calibration F2.")
                    print("=================================\n")
                    optimizer = OptimizerClass(self.NN.parameters(), lr=lr)
                elif i == 0:
                    print("\n=================================")
                    print("Calibration F1.")
                    print("=================================\n")
                    optimizer = OptimizerClass(self.NN.parameters(), lr=lr)

                def closure():
                    optimizer.zero_grad()
                    y = self.NN(k1_data_pts)
                    loss = self.loss_fn(y[i:3], y_data[i:3])
                    loss.backward()
                    grad = torch.cat(
                        [param.grad.view(-1) for param in self.NN.parameters()]
                    )
                    print("loss = ", loss.item())
                    print("div  = ", self.NN.div.abs().max().item())
                    # print('grad = ', grad.numpy())
                    # print( ('NN parameters = [' + ', '.join(['{}']*len(self.parameters)) + ']\n').format(*self.parameters))
                    # self.print()
                    self.kF_model_vals = y
                    self.plot(dynamic=True)
                    return loss

                # self.plot()
                for j in range(10):
                    optimizer.step(closure)
                    if closure() < 1.0e-3:
                        break

        elif self.NN.case_PowerSpectra == "C3Net":
            for i in (1,):
                optimizer = OptimizerClass(self.NN.parameters(), lr=lr)

                def closure():
                    optimizer.zero_grad()
                    y = self.NN(k1_data_pts)
                    loss = self.loss_fn(y, y_data)
                    print("Points:", y[1:3, 0], y_data[1:3, 0])
                    loss.backward()
                    grad = torch.cat(
                        [param.grad.view(-1) for param in self.NN.parameters()]
                    )
                    print("loss = ", loss.item())
                    print("grad = ", grad.numpy())
                    # print( ('NN parameters = [' + ', '.join(['{}']*len(self.parameters)) + ']\n').format(*self.parameters))
                    # self.print()
                    self.kF_model_vals = y
                    self.plot(dynamic=True)
                    return loss

                # self.plot()
                for j in range(10):  # len(self.k1_data_pts)):
                    # lr = lr/(j+1)
                    # k1_data_pts = self.k1_data_pts[:i+1]
                    # y_data = self.kF_data_vals[:,:i+1]
                    optimizer.step(closure)
                    if closure() < 5.0e-3:
                        break

        else:  ### Standard optimization
            alpha_reg = 0.001
            optimizer = OptimizerClass(self.NN.parameters(), lr=lr)

            def closure():
                optimizer.zero_grad()
                y = self.NN(k1_data_pts)
                pen = None  # 1.e1 * self.NN.div.abs().max()
                loss = self.loss_fn(y[:3], y_data[:3])
                if self.NN.case_PowerSpectra == "Corrector":
                    ### Corrector parameters regularization
                    Reg = torch.tensor(0.0)
                    for param in self.NN.Corrector.parameters():
                        Reg = Reg + param.norm(2) ** 2
                    Reg = Reg / len(list(self.NN.Corrector.parameters()))
                    print("reg = ", Reg.item())
                    loss = loss + alpha_reg * Reg
                loss.backward()
                grad = torch.cat(
                    [param.grad.view(-1) for param in self.NN.parameters()]
                )
                print(j)
                print("loss = ", loss.item())
                print("grad = ", grad.numpy())
                self.print()
                # print('div = ', self.NN.div.abs().max().item())
                self.kF_model_vals = y
                self.plot(dynamic=True)
                return loss

            for j in range(1000):
                optimizer.step(closure)
                if closure() < 1.0e-5:
                    break

        print("\n=================================")
        print("Calibration terminated.")
        print("=================================\n")

        # old_params = parameters_to_vector(self.NN.parameters())
        # for lr_j in lr * 0.1**np.arange(10):
        #     optimizer = method(self.NN.parameters(), lr=lr)
        #     for t in range(100):
        #         optimizer.step(closure)
        #         print(t)
        #         # if t%5==0: plot()
        #     current_params = parameters_to_vector(self.NN.parameters())
        #     if any(np.isnan(current_params.data.cpu().numpy())):
        #         print("Optimization diverged. Rolling back update...")
        #         vector_to_parameters(old_params, self.NN.parameters())
        #     else:
        #         break
        # # plot()

        # with torch.no_grad():
        #     self.parameters = parameters_to_vector(self.NN.parameters()).numpy()

        self.print()
        self.plot()
        return self.parameters

    ###------------------------------------------------
    ### Post-treatment and Export
    ###------------------------------------------------

    def print(self):
        print(
            (
                "Optimal NN parameters = ["
                + ", ".join(["{}"] * len(self.parameters))
                + "]\n"
            ).format(*self.parameters)
        )

    def plot(self, dynamic=False):
        k1 = self.k1_data_pts
        k = torch.stack([k1, 0 * k1, 0 * k1], dim=-1)

        if dynamic:
            ion()
        else:
            ioff()

        fg_plot_tau = True
        if not hasattr(self, "fig"):
            nrows = 1
            ncols = 2 if fg_plot_tau else 1
            self.fig, self.ax = subplots(
                nrows=nrows,
                ncols=ncols,
                num="Calibration",
                clear=True,
                figsize=[20, 10],
            )
            if not fg_plot_tau:
                self.ax = [self.ax]

            ### Subplot 1: One-point spectra
            self.ax[0].set_title("One-point spectra")
            self.lines_SP_model = [None] * (self.vdim + 1)
            self.lines_SP_data = [None] * (self.vdim + 1)
            self.kF_model_vals = self.NN(k1)
            for i in range(self.vdim):
                (self.lines_SP_model[i],) = self.ax[0].plot(
                    k1,
                    self.kF_model_vals[i].detach().numpy(),
                    "o-",
                    label=r"$F{0:d}$ model".format(i + 1),
                )
            for i in range(self.vdim):
                (self.lines_SP_data[i],) = self.ax[0].plot(
                    k1,
                    self.kF_data_vals[i].detach().numpy(),
                    "--",
                    label=r"$F{0:d}$ data".format(i + 1),
                )
            # self.lines_SP_model[self.vdim], = self.ax[0].plot(k1, -self.kF_model_vals[self.vdim].detach().numpy(), 'o-', label=r'$-F_{13}$ model')
            # self.lines_SP_data[self.vdim],  = self.ax[0].plot(k1, -self.kF_data_vals[self.vdim].detach().numpy(), '--', label=r'$-F_{13}$ data')
            self.ax[0].legend()
            self.ax[0].set_xscale("log")
            self.ax[0].set_yscale("log")
            self.ax[0].set_xlabel(r"$k_1$")
            self.ax[0].set_ylabel(r"$k_1 F_i$")
            self.ax[0].grid(which="both")
            self.ax[0].set_aspect(3 / 4)
            # self.ax[0].yaxis.set_minor_formatter(FormatStrFormatter())
            # self.ax[0].yaxis.set_major_formatter(FormatStrFormatter())
            # self.ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # self.ax[0].yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
            # self.ax[0].set_yticks(ticks=[0.01,0.02,0.05,0.1,0.2,0.5], minor=True)

            if fg_plot_tau:
                ### Subplot 2: Eddy Lifetime
                self.ax[1].set_title("Eddy liftime")
                self.tau_model = self.NN.EddyLifetime(k)
                k_norm = torch.norm(k, dim=-1).detach().numpy()
                # self.tau_ref = k_norm**(-2/3)
                self.tau_ref = 3.9 * MannEddyLifetime(
                    self.NN.Theta.abs().detach().numpy() * k_norm
                )
                (self.lines_LT_model,) = self.ax[1].plot(
                    k_norm,
                    self.tau_model.squeeze().detach().numpy(),
                    "-",
                    label=r"$\tau_{model}$",
                )
                # self.lines_LT_ref,   = self.ax[1].plot(k_norm, self.tau_ref.detach().numpy(),  '--', label=r'$\tau_{ref}=|k|^{-\frac{2}{3}}$')
                (self.lines_LT_ref,) = self.ax[1].plot(
                    k_norm, self.tau_ref, "--", label=r"$\tau_{ref}=$Mann"
                )
                self.ax[1].legend()
                # self.ax[1].set_aspect(3/4)
                self.ax[1].set_xscale("log")
                # self.ax[1].set_yscale('log')
                self.ax[1].set_xlabel(r"$k$")
                self.ax[1].set_ylabel(r"$\tau$")
                self.ax[1].grid(which="both")

        self.kF_model_vals = self.NN(k1)
        for i in range(self.vdim):
            self.lines_SP_model[i].set_ydata(self.kF_model_vals[i].detach().numpy())
        # self.lines_SP_model[self.vdim].set_ydata(-self.kF_model_vals[self.vdim].detach().numpy())
        self.ax[0].set_aspect(3 / 4)

        if fg_plot_tau:
            self.tau_model = self.NN.EddyLifetime(k)
            self.lines_LT_model.set_ydata(self.tau_model.detach().numpy())

        if dynamic:
            self.fig.gca().autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            plt.show()


############################################################################
############################################################################

if __name__ == "__main__":
    config = {
        "case_EddyLifetime": "Mann",  ### 'const', TwoThird', 'Mann', 'tauNet'
        "case_PowerSpectra": "RDT",  ### 'RDT', 'zetaNet', 'C3Net', 'Corrector'
        # 'input_size'        :   3,
        "hidden_layer_size": 3,
        "nModes": 10,  ### number of modes in the rational function in tauNet
        "degree": 1,  ### polynomial degree of input k^p in tauNet
        "tol": 1.0e-3,
        "init_with_noise": False,
        "flow_type": "shear",  ### 'shear', 'iso'
        "data_type": "Kaimal",  ### 'Kaimal', 'Simiu-Scanlan', 'Simiu-Yeo'
    }

    ### User-defined
    # L     = 1
    # Gamma = 4
    # sigma = 1

    ### Kaimmal
    L = 0.59
    Gamma = 3.9
    sigma = 3.2

    ### Simiu
    # L     = 0.79
    # Gamma = 3.8
    # sigma = 2.8

    L2 = L**2
    sigma = sigma * L ** (5 / 3) / (4 * np.pi)

    OPS = OnePointSpectra(**config)
    params = OPS.parameters

    params[:3] = [L, sigma, Gamma]
    OPS.parameters = params[: len(OPS.parameters)]
    OPS.print()

    ### Data
    k1_data_pts = np.logspace(-1, 2, 20)
    # k1_data_pts = np.array([0.005])
    # k1_data_pts = np.array([0.01])
    DataPoints = []
    for k1 in k1_data_pts:
        DataPoints.append((k1, 1))
    from drdmannturb.RandomFieldModule.Calibration.DataGenerator import \
        OnePointSpectraDataGenerator

    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data
    DataValues = Data[1]

    ####################################

    # ### Just plot
    kF = OPS(k1_data_pts)

    for i in range(3):
        plt.plot(k1_data_pts, kF[i], "o-", label=r"$F_{0:d}$ model".format(i + 1))
    # for i in range(3):
    #     # plt.plot(kF[i]/kF[i][0], 'o-', label=r'$F_{0:d}$ model'.format(i+1))
    #     plt.plot(k1_data_pts, kF[i], 'o-')#, label=r'$F_{0:d}$ model'.format(i+1))
    for i in range(3):
        plt.plot(
            k1_data_pts, DataValues[:, i, i], "--"
        )  # , label=r'$F_{0:d}$ data'.format(i+1))
    # plt.plot(k1_data_pts, -kF[3], 'o-', label=r'-$F_{13}$ model')
    # plt.plot(k1_data_pts, -DataValues[:,0,2], '--', label=r'$-F_{13}$ data')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$k_1$")
    plt.ylabel(r"$k_1 F(k_1)/u_\ast^2$")
    # plt.legend(['$u$ (model)', '$v$ (model)', '$w$ (model)', '$u$ (data)', '$v$ (data)', '$w$ (data)'])
    plt.legend()
    plt.grid(which="both")
    plt.show()

    ####################################

    ### Calibration
    opt_params = OPS.Calibrate(Data=Data, lr=1)  # , OptimizerClass=torch.optim.RMSprop)
