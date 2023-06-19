
import torch
import torch.nn as nn
from numpy import log

from .common import VKEnergySpectrum, MannEddyLifetime
from .PowerSpectraRDT import PowerSpectraRDT
from .tauNet import tauNet

"""
==================================================================================================================
Spectral Coherence class
==================================================================================================================
"""

class SpectralCoherence(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralCoherence, self).__init__()

        self.type_EddyLifetime = kwargs.get('type_EddyLifetime', 'TwoThird')
        self.type_PowerSpectra = kwargs.get('type_PowerSpectra', 'RDT')

        self.init_grids()
        self.init_parameters()

        if self.type_EddyLifetime == 'tauNet':
            self.tauNet = tauNet(**kwargs)

    ###-------------------------------------------

    def init_parameters(self):
        LengthScale = 0.7*42
        TimeScale = 3.9
        Magnitude = 1.0
        self.logLengthScale = nn.Parameter(torch.tensor(log(LengthScale), dtype=torch.float64))
        self.logTimeScale   = nn.Parameter(torch.tensor(log(TimeScale), dtype=torch.float64))
        self.logMagnitude   = nn.Parameter(torch.tensor(log(Magnitude), dtype=torch.float64))


    def update_scales(self):
        self.LengthScale = torch.exp(self.logLengthScale)
        self.TimeScale   = torch.exp(self.logTimeScale)
        self.Magnitude   = torch.exp(self.logMagnitude)
        return self.LengthScale.item(), self.TimeScale.item(), self.Magnitude.item()

    ###-------------------------------------------
    
    def init_grids(self):

        ### k2 grid
        p1, p2, N = -2, 2, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)**2
        grid_minus= -torch.flip(grid_plus, dims=[0])
        self.grid_k2 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        ### k3 grid
        p1, p2, N = -2, 2, 100
        grid_zero = torch.tensor([0], dtype=torch.float64)
        grid_plus = torch.logspace(p1, p2, N, dtype=torch.float64)**2
        grid_minus= -torch.flip(grid_plus, dims=[0])
        self.grid_k3 = torch.cat((grid_minus, grid_zero, grid_plus)).detach()

        self.meshgrid23 = torch.meshgrid(self.grid_k2, self.grid_k3)

    ###-------------------------------------------
    ### FORWARD MAP
    ###-------------------------------------------  

    def forward(self, k1_input, Delta_y_input, Delta_z_input):
        self.update_scales()
        self.k    = torch.stack(torch.meshgrid(k1_input, self.grid_k2, self.grid_k3), dim=-1)
        self.k123 = self.k[...,0], self.k[...,1], self.k[...,2]
        self.beta = self.EddyLifetime()
        self.k0   = self.k.clone()
        self.k0[...,2] = self.k[...,2] + self.beta * self.k[...,0]
        k0L = self.LengthScale * self.k0.norm(dim=-1)
        self.E0  = self.Magnitude * VKEnergySpectrum(k0L)
        self.Phi = self.PowerSpectra()
        # Phi11, Phi33, Phi13 = self.Phi[0], self.Phi[2], self.Phi[3]
        Phi11, Phi33, Phi13 = self.Phi[0], self.Phi[0], self.Phi[0]
        F1 = self.quad23(Phi11)
        F3 = self.quad23(Phi33)
        k2, k3 = self.k[...,1], self.k[...,2]
        Chi = torch.zeros([k1_input.numel(), Delta_y_input.numel(), Delta_z_input.numel()], dtype=torch.float64)
        for i, dy in enumerate(Delta_y_input):
            for j, dz in enumerate(Delta_z_input):
                Exponential = torch.exp(1j*(k2*dy + k3*dz))
                I = self.quad23(Phi13 * Exponential)
                den = torch.sqrt(F1 * F3)
                Chi[:,i,j] = torch.real(I / den)
        return Chi

    ###-------------------------------------------
    ### Auxilary methods
    ###-------------------------------------------  

    @torch.jit.export
    def EddyLifetime(self, k=None):
        if k is None:
            k = self.k
        else:
            self.update_scales()
        kL = self.LengthScale * k.norm(dim=-1)
        if self.type_EddyLifetime == 'const':
            tau = torch.ones_like(kL)
        elif self.type_EddyLifetime == 'Mann': ### uses numpy - can not be backpropagated !!
            tau = MannEddyLifetime(kL)
        elif self.type_EddyLifetime == 'TwoThird':
            tau = kL**(-2/3)
        elif self.type_EddyLifetime == 'tauNet':
            tau0 = self.InitialGuess_EddyLifetime(k.norm(dim=-1))
            tau  = tau0 + self.tauNet(k)
        else:
            raise Exception('Wrong EddyLifetime model !')
        return self.TimeScale * tau


    @torch.jit.export
    def InitialGuess_EddyLifetime(self, k_norm): 
        # tau0 = MannEddyLifetime(0.59*k_norm) 
        # tau0 = k_norm**(-2/3)
        tau0 = 0
        return tau0


    ###------------------------------------------- 

    @torch.jit.export
    def PowerSpectra(self):
        if self.type_PowerSpectra == 'RDT':
            Phi = PowerSpectraRDT(self.k, self.beta, self.E0)
        # elif self.type_PowerSpectra == 'Corrector':
        #     Corrector = self.Corrector(k)
        #     Phi = PowerSpectraCorr(self.k, beta, E0, Corrector)
        else:
            raise Exception('Wrong PowerSpectra model !')
        return Phi


    ###------------------------------------------- 

    ### Integration in k2 and k3
    @torch.jit.export
    def quad23(self, f):
        quad = torch.trapz(f,    x=self.k[...,2],   dim=-1)     ### integrate in k3
        quad = torch.trapz(quad, x=self.k[...,0,1], dim=-1)     ### integrate in k2 (just fix k3=0, since slices are idential in meshgrid)
        return quad


    ###------------------------------------------- 

    ### Divergence
    @torch.jit.export
    def get_div(self, Phi):
        k1, k2, k3 = self.freq
        Phi11, Phi22, Phi33, Phi13, Phi12, Phi23 = Phi
        div = torch.stack([ 
                k1*Phi11 + k2*Phi12 + k3*Phi13,
                k1*Phi12 + k2*Phi22 + k3*Phi23,
                k1*Phi13 + k2*Phi23 + k3*Phi33  
            ]) / (1/3 * (Phi11+Phi22+Phi33))
        return div