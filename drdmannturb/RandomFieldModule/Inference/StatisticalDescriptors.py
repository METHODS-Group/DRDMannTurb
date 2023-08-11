import copy
from math import *
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fft
import scipy.optimize
from pylab import *
from scipy.special import gamma

from ..PowerSpectra import MannPowerSpectrum
from ..utilities.common import MannEddyLifetime
from .FunctionExpansions import *

# =================================================================================================
#                               MODEL DESCRIPTORS
# =================================================================================================

###################################################################################################
#   Descriptor basic class
###################################################################################################


class BasicModelDescriptor:
    def __init__(self, ModelObj, **kwargs):
        self.ModelObj = ModelObj
        self.shape = list(self.ModelObj.grid_shape) + [3]
        self.verbose = kwargs.get("verbose", False)
        self.sigma = None

        L_inf = kwargs.get("L_inf", self.ModelObj.Covariance.corrlen)
        nExpansionTerms = kwargs.get("nExpansionTerms", 0)
        ExpansionType = kwargs.get("ExpansionType", None)
        self.ExpansionType = ExpansionType
        self.Anisotrop = kwargs.get("Anisotrop", False)

        if self.Anisotrop:
            if self.Anisotrop is 3:
                self.Expansion = [None] * 3
            else:
                self.Expansion = [None] * 2
        else:
            self.Expansion = [None]

        if ExpansionType is not None:
            for i in range(len(self.Expansion)):
                self.Expansion[i] = ExpansionType(L_inf=L_inf, nTerms=nExpansionTerms)
                # if ExpansionType is 'Exp':
                #     self.Expansion[i] = ExponentialExpansion(L_inf=L_inf, nTerms=nExpansionTerms)
                # elif ExpansionType is 'PowerExp':
                #     self.Expansion[i] = PowerExpExpansion(L_inf=L_inf)
                # elif ExpansionType is 'PolyExp':
                #     self.Expansion[i] = PolyExpExpansion(L_inf=L_inf, nTerms=nExpansionTerms)
                # elif ExpansionType is 'MP':
                #     self.Expansion[i] = MultiPoleExpansion(L_inf=L_inf, nTerms=nExpansionTerms)

    def __call__(self, p):
        return 0

    def default_parameters(self):
        return 0


###################################################################################################
#   Reynolds Stress
###################################################################################################


class DeterministicVarianceModelDescriptor(BasicModelDescriptor):
    def __init__(self, ModelObj, **kwargs):
        super().__init__(ModelObj, **kwargs)
        self.z_grid = kwargs.get("z_grid", None)

        mu = self.ModelObj.Correlate.factor
        L_infty = self.ModelObj.Correlate.corrlen
        NN = self.ModelObj.Correlate.Nd[0] * self.ModelObj.Correlate.Nd[1]
        self.sigma_infty = (
            2
            * mu**2
            * L_infty ** (2 / 3)
            * pi ** (3 / 2)
            * gamma(1 / 3)
            / (2 * (1 / 3 + 3 / 2) * gamma(1 / 3 + 3 / 2))
        )
        print("sigma_infty = ", self.sigma_infty)

    def __call__(self, p, loc, **kwargs):
        jac = kwargs.get("jac", False)
        doUW = kwargs.get("uw", False)
        doVV = kwargs.get("vv", False)
        doLS = kwargs.get("lengthscales", False)

        self.UpdateParameters(p)
        # if jac:
        #     grad_coef = self.Expansion.Gradient
        #     self.ModelObj.Correlate.fde_solve.reset_jac(grad_coef)

        # add "end point"
        # loc = np.append(loc,[0.25])
        loc = np.append(loc, [0.5])
        # loc = np.append(loc,[loc[-1]])

        # ## TODO: define nodes from locations
        # if self.z_grid is not None:
        #     h = np.diff(self.z_grid)
        #     Nodes_lower = np.zeros_like(loc, dtype=int)
        #     Weight_lower = np.zeros_like(loc)
        #     for i, iloc in enumerate(loc):
        #         Nodes_lower[i] = np.sum(1*(self.z_grid<=iloc))-1
        #         Weight_lower[i] = 1 - (iloc-self.z_grid[Nodes_lower[i]])/h[Nodes_lower[i]]
        #         pass
        # else:
        #     h = 1.0/self.shape[2]
        #     Nodes_lower = np.zeros_like(loc, dtype=int)
        #     Weight_lower = np.zeros_like(loc)
        #     for i, iloc in enumerate(loc):
        #         Nodes_lower[i] = iloc//h
        #         Weight_lower[i] = 1 - (iloc/h - iloc//h)

        # # Nodes =  list(range(self.shape[2]))
        # sigma = np.zeros([3,len(Nodes_lower)])
        # if doUW:
        #     sigma = np.zeros([4,len(Nodes_lower)])
        # if doLS:
        #     LS = np.zeros([2,2,len(Nodes_lower)])
        # if jac:
        #     grad_sigma = np.zeros([3,len(Nodes_lower),self.nPar])
        # if doVV:
        #     comp_list = [0,1,2]
        # else:
        #     comp_list = [0,2]
        # ### compute sigma
        # # for ind, iNode in enumerate(Nodes_lower):
        # #     for comp in range(3):
        # #         e = np.zeros(self.shape)
        # #         e[0,0,iNode,comp] = Weight_lower[ind]
        # #         e[0,0,iNode+1,comp] = 1-Weight_lower[ind]
        # #         Ainv_QT_e = self.ModelObj.Correlate(e, Robin_const=self.Robin_const, adjoint=True)
        # #         G2 = Ainv_QT_e**2
        # #         sigma[comp,ind] = np.mean(G2)
        # for ind, iNode in enumerate(Nodes_lower):
        #     for j in range(2):
        #         for comp in comp_list:
        #             e = np.zeros(self.shape)
        #             e[0,0,iNode+j,comp] = 1
        #             if jac:
        #                 Ainv_QT_e, grad_sum = self.ModelObj.Correlate(e, Robin_const=self.Robin_const, adjoint=True, jac=True, grad_coef=grad_coef)
        #                 G2 = math.pi**2
        #                 sigma[comp,ind] += np.sum(G2)*(j + (-1)**j * Weight_lower[ind])
        #                 grad_sigma[comp,ind,:] += grad_sum *(j + (-1)**j * Weight_lower[ind])
        #             else:
        #                 Ainv_QT_e = self.ModelObj.Correlate(e, Robin_const=self.Robin_const, adjoint=True)
        #                 G2 = Ainv_QT_e**2
        #                 sigma[comp,ind] += np.sum(G2)*(j + (-1)**j * Weight_lower[ind])
        #                 if doUW:
        #                     if comp==0: G2_uw = Ainv_QT_e
        #                     if comp==2: G2_uw *= Ainv_QT_e
        #                 if doLS:
        #                     ex = np.zeros(self.shape)
        #                     ey = np.zeros(self.shape)
        #                     ex[:,0,iNode+j,comp] = 1.0/self.shape[0]
        #                     ey[0,:,iNode+j,comp] = 1.0/self.shape[1]
        #                     Ainv_QT_ex = self.ModelObj.Correlate(ex, Robin_const=self.Robin_const, adjoint=True)
        #                     Ainv_QT_ey = self.ModelObj.Correlate(ey, Robin_const=self.Robin_const, adjoint=True)
        #                     G2x = Ainv_QT_ex*Ainv_QT_e
        #                     G2y = Ainv_QT_ey*Ainv_QT_e
        #                     if comp==0:
        #                         LS[0,0,ind] += np.sum(G2x)*(j + (-1)**j * Weight_lower[ind])
        #                         LS[0,1,ind] += np.sum(G2y)*(j + (-1)**j * Weight_lower[ind])
        #                     if comp==2:
        #                         LS[1,0,ind] += np.sum(G2x)*(j + (-1)**j * Weight_lower[ind])
        #                         LS[1,1,ind] += np.sum(G2y)*(j + (-1)**j * Weight_lower[ind])
        #         if doUW: sigma[3,ind] += np.sum(G2_uw)*(j + (-1)**j * Weight_lower[ind])

        ### NEW

        sigma = np.zeros([3, len(loc)])
        if doUW:
            sigma = np.zeros([4, len(loc)])
        if doLS:
            LS = np.zeros([3, len(loc)])

        # stress_comp_list = [0,1,2] if doVV else [0,2]
        # f1 = [None]*3
        # f2 = [None]*3
        # grid = self.ModelObj.Correlate.fde_solve.grid[:]

        # k1, k2, _ = np.meshgrid(*self.ModelObj.Correlate.Frequencies, indexing='ij')

        # for iloc, z in enumerate(loc):
        #     ind = np.sum(1*(grid<=z))-1
        #     h = grid[ind+1]-grid[ind]
        #     w = 1 - (z-grid[ind])/h
        #     for stress_comp in stress_comp_list:
        #         # f1[stress_comp] = self.ModelObj.Correlate.compute_Adjoint(stress_comp=stress_comp, z=grid[ind],   Robin_const=self.Robin_const)
        #         # f2[stress_comp] = self.ModelObj.Correlate.compute_Adjoint(stress_comp=stress_comp, z=grid[ind+1], Robin_const=self.Robin_const)
        #         e = np.zeros(self.shape, dtype=np.complex)
        #         e[:,:,ind,stress_comp] = 1
        #         f1[stress_comp] = self.ModelObj.Correlate(e, Robin_const=self.Robin_const, adjoint=True)
        #         # e[0,0,ind,stress_comp] = 1
        #         # f1[stress_comp] = self.ModelObj.Correlate.compute_FFT_Adjoint(e, Robin_const=self.Robin_const)
        #         e = np.zeros(self.shape, dtype=np.complex)
        #         e[:,:,ind+1,stress_comp] = 1
        #         f2[stress_comp] = self.ModelObj.Correlate(e, Robin_const=self.Robin_const, adjoint=True)
        #         # e[0,0,ind+1,stress_comp] = 1
        #         # f2[stress_comp] = self.ModelObj.Correlate.compute_FFT_Adjoint(e, Robin_const=self.Robin_const)

        #     f1[0][...,0] *= k2; f2[0][...,0] *= k2
        #     # f1[2][...,1] *= k1; f2[2][...,1] *= k1

        #     sigma[0,iloc] = np.sum(np.abs(f1[0])**2) * w + np.sum(np.abs(f2[0])**2) * (1-w)
        #     sigma[2,iloc] = np.sum(np.abs(f1[2])**2) * w + np.sum(np.abs(f2[2])**2) * (1-w)
        #     if doVV: sigma[1,iloc] = np.sum(np.abs(f1[1])**2) * w + np.sum(np.abs(f2[1])**2) * (1-w)
        #     if doUW: sigma[3,iloc] = np.sum(f1[0]*f1[2]) * w + np.sum(f2[0]*f2[2]) * (1-w)

        #     if doLS:
        #         LS[0,0,iloc] = np.sum(np.abs(f1[0][0,...])**2) * w + np.sum(np.abs(f2[0][0,...])**2) * (1-w)
        #         LS[0,1,iloc] = np.sum(np.abs(f1[0][:,0,...])**2) * w + np.sum(np.abs(f2[0][:,0,...])**2) * (1-w)
        #         LS[1,0,iloc] = np.sum(np.abs(f1[2][0,...])**2) * w + np.sum(np.abs(f2[2][0,...])**2) * (1-w)
        #         LS[1,1,iloc] = np.sum(np.abs(f1[2][:,0,...])**2) * w + np.sum(np.abs(f2[2][:,0,...])**2) * (1-w)
        #         LS[:,:,:]  *= self.ModelObj.Correlate.Nd[0]*self.ModelObj.Correlate.Nd[1]

        ## Polar integration
        # r = 1.e4 * np.linspace(0,1,self.ModelObj.Correlate.Nd[1])**5 ### grid 4
        r = (
            1.0e3 * np.linspace(0, 1, self.ModelObj.Correlate.Nd[1]) ** 3
        )  ### oprimalfor grid 7
        k3 = (2 * pi / self.ModelObj.Correlate.L[2]) * (
            self.ModelObj.Correlate.Nd[2] * fft.fftfreq(self.ModelObj.Correlate.Nd[2])
        )
        rr, _ = np.array(list(np.meshgrid(*[r, k3], indexing="ij")))
        r = rr[:, 0]
        r2 = r**2
        dr = r[1:] - r[:-1]

        def Quad(f):
            return dr.dot(f[1:] + f[:-1]) / 2

        def norm_z(f):
            SMf = np.apply_along_axis(
                self.ModelObj.Correlate.fde_solve.apply_sqrtMass, -1, f
            )
            return np.sum(np.abs(SMf) ** 2, axis=-1)

        f1 = [None] * 3
        f2 = [None] * 3
        grid = self.ModelObj.Correlate.fde_solve.grid[:]
        for iloc, z in enumerate(loc):
            ind = np.sum(1 * (grid <= z)) - 1
            h = grid[ind + 1] - grid[ind]
            w = 1 - (z - grid[ind]) / h  ### weight
            e = np.zeros(len(grid))

            ### Node n
            e[ind] = 1

            f1[
                2
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                e, r, component=0, Robin_const=self.Robin_const
            )
            f1[2] = r**2 * norm_z(f1[2])

            f1[
                1
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                e, r, component=2, Robin_const=self.Robin_const
            )
            f1[1] = r**2 * norm_z(f1[1])

            De = self.ModelObj.Correlate.Dz(e, adjoint=True)
            f1[
                0
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                De, r, component=0, Robin_const=self.Robin_const
            )
            f1[0] = norm_z(f1[0])

            ### Node n+1
            e[ind], e[ind + 1] = 0, 1

            f2[
                2
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                e, r, component=0, Robin_const=self.Robin_const
            )
            f2[2] = r**2 * norm_z(f2[2])

            f2[
                1
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                e, r, component=2, Robin_const=self.Robin_const
            )
            f2[1] = r**2 * norm_z(f2[1])

            De = self.ModelObj.Correlate.Dz(e, adjoint=True)
            f2[
                0
            ] = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier_polar(
                De, r, component=0, Robin_const=self.Robin_const
            )
            f2[0] = norm_z(f2[0])

            ### <ww>
            WW1, WW2 = 2 * pi * Quad(f1[2] * r), 2 * pi * Quad(f2[2] * r)
            sigma[2, iloc] = WW1 * w + WW2 * (1 - w)

            ### <uu>
            UU1, UU2 = 2 * pi * Quad((f1[0] + f1[1] / 2) * r), 2 * pi * Quad(
                (f2[0] + f2[1] / 2) * r
            )
            sigma[0, iloc] = UU1 * w + UU2 * (1 - w)

            if doVV:
                sigma[1, iloc] = np.sum(np.abs(f1[1]) ** 2) * w + np.sum(
                    np.abs(f2[1]) ** 2
                ) * (1 - w)
            if doUW:
                sigma[3, iloc] = np.sum(f1[0] * f1[2]) * w + np.sum(f2[0] * f2[2]) * (
                    1 - w
                )

            if doLS:
                LS[0, iloc] = Quad(f1[0] + f1[1]) / UU1 * w + Quad(
                    f2[0] + f2[1]
                ) / UU2 * (1 - w)
                LS[1, iloc] = Quad(f1[0]) / UU1 * w + Quad(f2[0]) / UU2 * (1 - w)
                LS[2, iloc] = Quad(f1[2]) / WW1 * w + Quad(f2[2]) / WW2 * (1 - w)

                # LS[0,iloc] /= sigma[0,iloc]
                # LS[1,iloc] /= sigma[0,iloc]
                # LS[2,iloc] /= sigma[2,iloc]

        # sigma[:,:] *= pi**2 #self.ModelObj.Correlate.Nd[0]*self.ModelObj.Correlate.Nd[1]
        # if doLS:
        #     LS[:,:] *= 2 #self.ModelObj.Correlate.Nd[0]

        print("Sigma[-1] =", sigma[0, -1])
        print("Sigma[0] =", sigma[0, 0])
        # self.sigma_infty = sigma[0,0]/2
        # sigma[:,-1] = self.sigma_infty

        self.sigma = np.zeros([sigma.shape[0], sigma.shape[1] - 1])
        self.sigma[0, :] = sigma[0, :-1] / sigma[0, -1]
        self.sigma[2, :] = sigma[2, :-1] / sigma[2, -1]
        if doVV:
            self.sigma[1, :] = sigma[1, :-1] / sigma[1, -1]
        else:
            self.sigma[1, :] = self.sigma[0, :]
        if doUW:
            self.sigma[3, :] = sigma[3, :-1] / sigma[3, -1]

        # if jac:
        #     self.grad_sigma = np.zeros([3,len(Nodes_lower)-1,self.nPar])
        #     self.grad_sigma[0,:,:] = grad_sigma[0,:-1,:]/sigma[0,-1]
        #     self.grad_sigma[1,:,:] = grad_sigma[1,:-1,:]/sigma[1,-1]
        #     self.grad_sigma[2,:,:] = grad_sigma[2,:-1,:]/sigma[2,-1]

        if doLS:
            self.LS = LS[:, :-1] / LS[0, -1]
            return self.sigma, self.LS

        if not jac:
            return self.sigma
        else:
            return self.sigma, self.grad_sigma

    def UpdateParameters(self, p):
        params = iter(p)
        for ExpansionI in self.Expansion:
            ExpansionI.update([next(params) for i in range(ExpansionI.nPars)])
        self.ExtraParams = [param for param in params]
        self.nPar = len(p)

        if self.Anisotrop:
            coef = [self.Expansion[0], self.Expansion[0], self.Expansion[1]]
        else:
            coef = self.Expansion[0]
        self.ModelObj.Correlate.fde_solve.reset_parameters(coef=coef)

        self.Robin_const = self.ExtraParams[0]
        if self.ExtraParams[0] is not np.infty:
            self.Robin_const *= self.Expansion[0].L_inf

    def Solve(self, noise, p, adjoint=False):
        self.UpdateParameters(p)
        field = self.ModelObj.Correlate(
            noise, Robin_const=self.Robin_const, adjoint=adjoint
        )
        return field

    def default_parameters(self):
        return 0


###################################################################################################
#   Rapid distortion one-point spectra
###################################################################################################


class RD1PointSpectraModelDescriptor(BasicModelDescriptor):
    def __init__(self, ModelObj, **kwargs):
        super().__init__(ModelObj, **kwargs)
        self.k1_grid = kwargs.get("k1_grid", self.ModelObj.Correlate.Frequencies[0])

    def __call__(self, p, loc, **kwargs):
        self.UpdateParameters(p)
        k1z = np.geomspace(0.1, 100, 20)

        N = self.ModelObj.Correlate.Nd[1]
        k1 = self.k1_grid
        # k2 = (2*pi)*  np.r_[0, (1.5**np.arange(60)).astype(int)]
        # k2 = (2*pi)*(N*fft.fftfreq(N)) # self.ModelObj.Correlate.Frequencies[1]#**3 ### oprimalfor grid 7
        # k2 = (2*pi)*np.linspace(0,1,100)**3 * 1.e3
        # k2 = (2*pi)*np.r_[0, np.geomspace(1,10000,100)]
        # k2 = (2*pi)*np.r_[0, np.logspace(-4,4,100)]
        k2 = (2 * pi) * np.r_[-np.logspace(-2, 4, 20)[::-1], 0, np.logspace(-2, 4, 20)]
        k3 = self.ModelObj.Correlate.Frequencies[2]

        grid_z = self.ModelObj.Correlate.fde_solve.grid[
            :
        ]  ### grid in z (only uniform !!)

        F1 = np.zeros([len(k1), len(loc)])
        F2 = np.zeros_like(F1)
        F3 = np.zeros_like(F1)

        ### Distorted noise covariance
        D_tau = self.ModelObj.Correlate.distorted_noise_covariance(k1, k2, k3, self.tau)

        ## Integration
        dx = k2[1:] - k2[:-1]

        def Quad(f):  ### integration over k2
            # f = np.sum(f, axis=2)
            return (f[:, 1:] + f[:, :-1]) @ dx / 2

        def sca_z(f, g):
            Mf = np.apply_along_axis(
                self.ModelObj.Correlate.fde_solve.apply_Mass, -1, f
            )
            y = np.real(np.conj(g) * Mf)
            # y /= (self.ModelObj.Correlate.L[2] / self.ModelObj.Correlate.Nd[2])  ### normalize due to the discrete noise in x and y
            return np.sum(y, axis=-1)

        adjVecPot = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier
        args = {"Robin_const": self.Robin_const, "tau": self.tau}

        # k1, k2 = np.meshgrid(k1, k2, indexing='ij')

        f1 = [None] * 2
        f2 = [None] * 2
        f3 = [None] * 2
        # f = [[None]*6]*2
        # f = [[None]*2]*2
        for iloc, z in enumerate(loc):
            if z < grid_z.max():
                ind = np.sum(1 * (grid_z <= z)) - 1
            elif z == grid_z.max():
                ind = grid_z.size - 1
            else:
                raise Exception("Data point is out of z-grid !")
            h = grid_z[ind + 1] - grid_z[ind]
            w = 1 - (z - grid_z[ind]) / h  ### weight
            e = np.zeros(len(grid_z))

            # k1 = k1z/z #/(2*pi)

            # ### Distorted noise covariance
            # D_tau = self.ModelObj.Correlate.distorted_noise_covariance(k1, k2, k3, self.tau)

            for node in range(2):
                e[ind + node] = 1
                e[ind + 1 - node] = 0
                De = self.ModelObj.Correlate.Dz(e, adjoint=True)

                # p  = adjVecPot( e, k1, k2, component=0, **args)
                # p3 = adjVecPot( e, k1, k2, component=2, **args)
                # Dp = adjVecPot(De, k1, k2, component=0, **args)
                # # f[node][2] = adjVecPot(e, k1, k2, component=2, **args)
                # # f[node][3] = adjVecPot(e, k1, k2, component=0, **args)
                # # f[node][4] = adjVecPot(e, k1, k2, component=0, **args)
                # # f[node][5] = adjVecPot(e, k1, k2, component=0, **args)

                # # p  = fft.fft(p,  axis=-1)
                # # p3 = fft.fft(p3, axis=-1)
                # # Dp = fft.fft(Dp, axis=-1)

                # # p1 = p + D_tau[0]*p3
                # # p2 = p + D_tau[1]*p3
                # # Dp1 = Dp + D_tau[0]*p3
                # # Dp2 = Dp + D_tau[1]*p3
                # # p3 = D_tau[2]*p3

                # # p1 = fft.ifft(p1,  axis=-1)
                # # p2 = fft.ifft(p2,  axis=-1)
                # # Dp1 = fft.ifft(Dp1,  axis=-1)
                # # Dp2 = fft.ifft(Dp2,  axis=-1)
                # # p3 = fft.ifft(p3,  axis=-1)

                # # G1p3 = p3 #fft.ifft(D_tau[0]*fft.fft(p3, axis=-1), axis=-1)
                # # G2p3 = p3 #fft.ifft(D_tau[1]*fft.fft(p3, axis=-1), axis=-1)
                # # G3p3 = p3 #fft.ifft(D_tau[2]*fft.fft(p3, axis=-1), axis=-1)

                # G1p3 = fft.ifft(D_tau[0]*fft.fft(p3, axis=-1), axis=-1)
                # G2p3 = fft.ifft(D_tau[1]*fft.fft(p3, axis=-1), axis=-1)
                # G3p3 = fft.ifft(D_tau[2]*fft.fft(p3, axis=-1), axis=-1)

                # k2p  = 1j*k2[None,:,None]*p
                # k1p  = 1j*k1[:,None,None]*p
                # k1G1p3 = 1j*k1[:,None,None]*G1p3
                # k2G1p3 = 1j*k2[None,:,None]*G1p3
                # k1G2p3 = 1j*k1[:,None,None]*G2p3
                # k2G2p3 = 1j*k2[None,:,None]*G2p3
                # k1G3p3 = 1j*k1[:,None,None]*G3p3
                # k2G3p3 = 1j*k2[None,:,None]*G3p3

                # k1p3_sqr = sca_z(k1G1p3, k1G1p3) + sca_z(k1G2p3, k1G2p3) + sca_z(k1G3p3, k1G3p3)
                # k2p3_sqr = sca_z(k2G1p3, k2G1p3) + sca_z(k2G2p3, k2G2p3) + sca_z(k2G3p3, k2G3p3)

                # f1[node] = k2p3_sqr - 2*sca_z(k2G2p3, Dp) + sca_z(Dp, Dp)
                # f2[node] = sca_z(Dp, Dp) - 2*sca_z(k1G1p3, Dp) + k1p3_sqr
                # f3[node] = sca_z(k2p, k2p) + sca_z(k1p, k1p)

                p1 = adjVecPot(e, k1, k2, component=0, **args)
                p2 = adjVecPot(e, k1, k2, component=1, **args)
                p3 = adjVecPot(e, k1, k2, component=2, **args)
                Dp1 = adjVecPot(De, k1, k2, component=0, **args)
                Dp2 = adjVecPot(De, k1, k2, component=1, **args)

                G1p3 = fft.ifft(D_tau[0] * fft.fft(p3, axis=-1), axis=-1)
                G2p3 = fft.ifft(D_tau[1] * fft.fft(p3, axis=-1), axis=-1)
                G3p3 = fft.ifft(D_tau[2] * fft.fft(p3, axis=-1), axis=-1)

                k2p1 = 1j * k2[None, :, None] * p1
                k1p2 = 1j * k1[:, None, None] * p2
                k1G1p3 = 1j * k1[:, None, None] * G1p3
                k2G1p3 = 1j * k2[None, :, None] * G1p3
                k1G2p3 = 1j * k1[:, None, None] * G2p3
                k2G2p3 = 1j * k2[None, :, None] * G2p3
                k1G3p3 = 1j * k1[:, None, None] * G3p3
                k2G3p3 = 1j * k2[None, :, None] * G3p3

                k1p3_sqr = (
                    sca_z(k1G1p3, k1G1p3)
                    + sca_z(k1G2p3, k1G2p3)
                    + sca_z(k1G3p3, k1G3p3)
                )
                k2p3_sqr = (
                    sca_z(k2G1p3, k2G1p3)
                    + sca_z(k2G2p3, k2G2p3)
                    + sca_z(k2G3p3, k2G3p3)
                )

                f1[node] = k2p3_sqr - 2 * sca_z(k2G2p3, Dp2) + sca_z(Dp2, Dp2)
                f2[node] = sca_z(Dp1, Dp1) - 2 * sca_z(k1G1p3, Dp1) + k1p3_sqr
                f3[node] = sca_z(k2p1, k2p1) + sca_z(k1p2, k1p2)

            F1[:, iloc] = k1 * Quad(f1[0] * w + f1[1] * (1 - w))
            F2[:, iloc] = k1 * Quad(f2[0] * w + f2[1] * (1 - w))
            F3[:, iloc] = k1 * Quad(f3[0] * w + f3[1] * (1 - w))

        self.OnePointSpectra = np.array([F1, F2, F3])
        self.OnePointSpectra *= self.ModelObj.Correlate.factor**2 * (z + 1.0e-6) ** (
            -2 / 3
        )

        return self.OnePointSpectra

    def UpdateParameters(self, p):
        params = iter(p)
        for ExpansionI in self.Expansion:
            # ExpansionI.update([ next(params) for i in range(ExpansionI.nPars) ])
            # ExpansionI.update([ 0 for i in range(ExpansionI.nPars) ])
            ExpansionI.update([next(params), next(params)])
        # self.ExtraParams = [ param for param in params ]
        self.nPar = len(p)

        ### Lengthscales
        # L_inf = np.abs(next(params))
        # for i in range(3):
        #     self.Expansion[i].update_L_inf(L_inf)
        L = self.Expansion[0].L_inf
        L1 = self.Expansion[0].L_inf * self.Expansion[0].a
        L2 = self.Expansion[1].L_inf * self.Expansion[1].a

        ### RC
        self.Robin_const = (next(params)) ** 2
        # self.Robin_const = np.infty
        if not np.isinf(self.Robin_const):
            self.Robin_const *= L

        ### Distortion time
        self.MannGamma = next(params)
        self.MannL = next(params)
        # # self.tau = lambda k1, k2: self.MannGamma * MannEddyLifetime(np.sqrt((k1*L1)**2+(k2*L2)**2))
        self.tau = lambda k1, k2, k3: self.MannGamma * MannEddyLifetime(
            np.sqrt(k1**2 + k2**2) * self.MannL
        )
        # # self.tauFactor = next(params)
        # # self.tauPower  = -(next(params))**2
        # # self.tau = lambda k1, k2, k3: self.tauFactor * (k1**2+k2**2)**self.tauPower
        # self.tau = next(params)

        ### Normalization constant
        self.factor = next(params)
        self.ModelObj.Correlate.factor = (
            self.factor
        )  # * (self.ModelObj.Correlate.z_grid+1.e-6)**(-2/3)

        ### Update coefs
        if self.Anisotrop:
            coef = [ExpansionI for ExpansionI in self.Expansion]
        else:
            coef = self.Expansion[0]
        self.ModelObj.Correlate.fde_solve.reset_parameters(coef=coef)


###################################################################################################
#   Rapid distortion Reynolds Stress
###################################################################################################


class RDReynoldsStressModelDescriptor(BasicModelDescriptor):
    def __init__(self, ModelObj, **kwargs):
        super().__init__(ModelObj, **kwargs)
        self.k1_grid = kwargs.get("k1_grid", self.ModelObj.Correlate.Frequencies[0])

    def __call__(self, p, loc, **kwargs):
        self.UpdateParameters(p)
        k1 = self.k1_grid
        # k2 = k1
        # k2 = (2*pi)*  np.r_[0, (1.5**np.arange(60)).astype(int)]
        # k2 = (2*pi)*(N*fft.fftfreq(N)) # self.ModelObj.Correlate.Frequencies[1]#**3 ### oprimalfor grid 7
        # k2 = (2*pi)*np.linspace(0,1,200)**3 * 1.e3
        # k2 = (2*pi)*np.r_[0, np.geomspace(1,10000,100)]
        k2 = (2 * pi) * np.r_[0, np.logspace(-4, 4, 100)]
        k3 = self.ModelObj.Correlate.Frequencies[2]

        grid_z = self.ModelObj.Correlate.fde_solve.grid[
            :
        ]  ### grid in z (only uniform !!)

        uu = np.zeros([len(loc)])
        vv = np.zeros_like(uu)
        ww = np.zeros_like(uu)
        uw = np.zeros_like(uu)

        ### Distorted noise covariance
        D_tau = self.ModelObj.Correlate.distorted_noise_covariance(k1, k2, k3, self.tau)

        ## Integration
        dx = k1[1:] - k1[:-1]
        dy = k2[1:] - k2[:-1]

        def Quad(f):  ### integration over k1 and k2
            tmp = (f[:, 1:] + f[:, :-1]) @ dy / 2
            return dx.dot(tmp[1:] + tmp[:-1]) / 2

        def sca_z(f, g):
            Mf = np.apply_along_axis(
                self.ModelObj.Correlate.fde_solve.apply_Mass, -1, f
            )
            # y = np.real(np.conj(g)*Mf)
            y = np.conj(g) * Mf
            # y /= (self.ModelObj.Correlate.L[2] / self.ModelObj.Correlate.Nd[2])  ### normalize due to the discrete noise in x and y
            return np.sum(y, axis=-1)

        adjVecPot = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier
        args = {"Robin_const": self.Robin_const, "tau": self.tau}

        f1 = [None] * 2
        f2 = [None] * 2
        f3 = [None] * 2
        f4 = [None] * 2
        for iloc, z in enumerate(loc):
            if z < grid_z.max():
                ind = np.sum(1 * (grid_z <= z)) - 1
            elif z == grid_z.max():
                ind = grid_z.size - 1
            else:
                raise Exception("Data point is out of z-grid !")
            h = grid_z[ind + 1] - grid_z[ind]
            w = 1 - (z - grid_z[ind]) / h  ### weight
            e = np.zeros(len(grid_z))

            for node in range(2):
                e[ind + node] = 1
                e[ind + 1 - node] = 0
                De = self.ModelObj.Correlate.Dz(e, adjoint=True)

                p1 = adjVecPot(e, k1, k2, component=0, **args)
                p2 = p1
                # p2  = adjVecPot( e, k1, k2, component=1, **args)
                p3 = adjVecPot(e, k1, k2, component=2, **args)
                Dp1 = adjVecPot(De, k1, k2, component=0, **args)
                Dp2 = Dp1
                # Dp2 = adjVecPot(De, k1, k2, component=1, **args)

                G1p3 = fft.ifft(D_tau[0] * fft.fft(p3, axis=-1), axis=-1)
                G2p3 = fft.ifft(D_tau[1] * fft.fft(p3, axis=-1), axis=-1)
                G3p3 = fft.ifft(D_tau[2] * fft.fft(p3, axis=-1), axis=-1)

                k2p1 = 1j * k2[None, :, None] * p1
                k1p2 = 1j * k1[:, None, None] * p2
                k1G1p3 = 1j * k1[:, None, None] * G1p3
                k2G1p3 = 1j * k2[None, :, None] * G1p3
                k1G2p3 = 1j * k1[:, None, None] * G2p3
                k2G2p3 = 1j * k2[None, :, None] * G2p3
                k1G3p3 = 1j * k1[:, None, None] * G3p3
                k2G3p3 = 1j * k2[None, :, None] * G3p3

                k1p3_sqr = (
                    sca_z(k1G1p3, k1G1p3)
                    + sca_z(k1G2p3, k1G2p3)
                    + sca_z(k1G3p3, k1G3p3)
                )
                k2p3_sqr = (
                    sca_z(k2G1p3, k2G1p3)
                    + sca_z(k2G2p3, k2G2p3)
                    + sca_z(k2G3p3, k2G3p3)
                )

                f1[node] = k2p3_sqr - 2 * sca_z(k2G2p3, Dp2) + sca_z(Dp2, Dp2)
                f2[node] = k1p3_sqr - 2 * sca_z(k1G1p3, Dp1) + sca_z(Dp1, Dp1)
                f3[node] = sca_z(k2p1, k2p1) + sca_z(k1p2, k1p2)
                f4[node] = sca_z(k2G2p3, k1p2) - sca_z(
                    k2G1p3, k2p1
                )  # - sca_z(Dp2, k1p2) #+ sca_z(Dp2, k2p1) # - p1 and p2 independent. Moreover, -sca_z(Dp2, k1p2) = sigma(uw)_iso = 0 (?)

                f1[node] = np.real(f1[node])
                f2[node] = np.real(f2[node])
                f3[node] = np.real(f3[node])
                f4[node] = np.real(f4[node])

            uu[iloc] = Quad(f1[0] * w + f1[1] * (1 - w))
            vv[iloc] = Quad(f2[0] * w + f2[1] * (1 - w))
            ww[iloc] = Quad(f3[0] * w + f3[1] * (1 - w))
            uw[iloc] = Quad(f4[0] * w + f4[1] * (1 - w))

        self.ReynoldsStresses = np.array([uu, vv, ww, uw])
        self.ReynoldsStresses *= self.ModelObj.Correlate.factor**2 * (z + 1.0e-6) ** (
            -2 / 3
        )

        return self.ReynoldsStresses

    def UpdateParameters(self, p):
        params = iter(p)
        for ExpansionI in self.Expansion:
            # ExpansionI.update([ next(params) for i in range(ExpansionI.nPars) ])
            # ExpansionI.update([ 0 for i in range(ExpansionI.nPars) ])
            ExpansionI.update([next(params), next(params)])
        # self.ExtraParams = [ param for param in params ]
        self.nPar = len(p)

        ### Lengthscales
        # L_inf = np.abs(next(params))
        # for i in range(3):
        #     self.Expansion[i].update_L_inf(L_inf)

        ### RC
        self.Robin_const = np.abs(next(params))
        # self.Robin_const = np.infty
        if not np.isinf(self.Robin_const):
            self.Robin_const *= self.Expansion[0].L_inf
        L = self.Expansion[0].L_inf
        L1 = self.Expansion[0].L_inf * self.Expansion[0].a
        L2 = self.Expansion[1].L_inf * self.Expansion[1].a
        L3 = self.Expansion[2].L_inf * self.Expansion[2].a

        ### Distortion time
        self.MannGamma = next(params)
        self.tau = lambda k1, k2, z: self.MannGamma * MannEddyLifetime(k1)
        # self.tau = lambda k1, k2, z: self.MannGamma * MannEddyLifetime(np.sqrt((k1*L1)**2+(k2*L2)**2))
        # self.tau = lambda k1, k2, k3: self.MannGamma * MannEddyLifetime(np.sqrt((k1*L1)**2+(k2*L2)**2 + 0*(k3*L3)**2))

        ### Normalization constant
        self.factor = np.abs(next(params))
        self.ModelObj.Correlate.factor = (
            self.factor
        )  # * (self.ModelObj.Correlate.z_grid+1.e-6)**(-2/3)

        ### Update coefs
        if self.Anisotrop:
            coef = [ExpansionI for ExpansionI in self.Expansion]
        else:
            coef = self.Expansion[0]
        self.ModelObj.Correlate.fde_solve.reset_parameters(coef=coef)


###################################################################################################
#   Mann's one-point spectra
###################################################################################################


class MannModelDescriptor(BasicModelDescriptor):
    def __init__(self, ModelObj, **kwargs):
        super().__init__(ModelObj, **kwargs)
        self.k1_grid = kwargs.get("k1_grid", self.ModelObj.Correlate.Frequencies[0])

    def __call__(self, p, loc, **kwargs):
        doRS = kwargs.get("ReynoldsStress", False)

        self.UpdateParameters(p)

        z_ref = loc[0]

        k1 = self.k1_grid
        # k2 = (2*pi)*np.logspace(-4,4,100)
        # k2 = (2*pi)*np.r_[0, np.logspace(-4,4,100)]

        log_k_min = 3
        log_k_max = 4
        n_k = 100
        k3 = (2 * pi) * np.r_[
            -np.logspace(-log_k_min, log_k_max, n_k)[::-1],
            0,
            np.logspace(-log_k_min, log_k_max, n_k),
        ]
        # grid = np.linspace(0, 1, n_k) **5
        # k3 = (2*pi)*np.r_[ -grid[::-1], 0, grid ] * 1.e1

        # log_k_min = 4
        # log_k_max = 5
        # n_k = 100
        # # k2 = (2*pi)*np.r_[-np.logspace(-log_k_min,log_k_max,n_k)[::-1], 0, np.logspace(-log_k_min,log_k_max,n_k)]
        # k2 =np.linspace(-100,100,100)
        # # k2 = (2*pi)*np.r_[0, np.logspace(-log_k_min,log_k_max,n_k)]
        # # k2 = (2*pi)*np.r_[0, np.logspace(-log_k_min,log_k_max,n_k)]
        # # k2 = (2*pi)*np.r_[-np.logspace(0,log_k_max,n_k)[::-1], np.logspace(-log_k_min,-1,n_k)[::-1], 0, np.logspace(-log_k_min,-1,n_k), np.logspace(0,log_k_max,n_k)]
        k2 = k3

        # N = 200
        # Nd = [N, N, N]
        # k2 = 2*pi*Nd[1]*fft.fftfreq(Nd[1])
        # k3 = 2*pi*Nd[2]*fft.fftfreq(Nd[2])

        ## Integration
        dx1 = k1[1:] - k1[:-1]
        dx2 = k2[1:] - k2[:-1]
        dx3 = k3[1:] - k3[:-1]

        def Quad23(f):
            I = np.sum((f[..., 1:] + f[..., :-1]) * dx3, axis=-1) / 2
            I = np.sum((I[:, 1:] + I[:, :-1]) * dx2, axis=-1) / 2
            return I

        def Quad(f):
            I = Quad23(f)
            I = np.sum((I[1:] + I[:-1]) * dx1, axis=-1) / 2
            return I

        k = np.array(list(np.meshgrid(k1, k2, k3, indexing="ij")))
        PowerSpectra = MannPowerSpectrum(
            k, L=self.MannL, Gamma=self.MannGamma, factor=self.MannFactor
        )

        if doRS:  ### Reynolds stresses
            sigma11 = Quad(PowerSpectra[0, 0, ...])
            sigma22 = Quad(PowerSpectra[1, 1, ...])
            sigma33 = Quad(PowerSpectra[2, 2, ...])
            sigma13 = Quad(PowerSpectra[0, 2, ...])
            self.ReynoldsStress = np.array([sigma11, sigma22, sigma33, sigma13])
            return self.ReynoldsStress

        else:  ### 1-point spectra
            F1 = k1 * Quad23(PowerSpectra[0, 0, ...])
            F2 = k1 * Quad23(PowerSpectra[1, 1, ...])
            F3 = k1 * Quad23(PowerSpectra[2, 2, ...])
            self.OnePointSpectra = np.array([F1, F2, F3]).reshape([3, -1, 1])
            return self.OnePointSpectra

    def UpdateParameters(self, p):
        params = iter(p)

        self.MannL = next(params)
        self.MannGamma = next(params)
        self.MannFactor = next(params)


###################################################################################################
###################################################################################################


class VarianceModelDescriptor(BasicModelDescriptor):
    def __init__(self, ModelObj, **kwargs):
        super().__init__(ModelObj, **kwargs)
        self.sigma = np.zeros([3, self.shape[2]])
        self.nsamples = kwargs.get("nsamples", 100)
        self.tol = kwargs.get("tol", 1.0e-2)

    def __call__(self, p):
        params = iter(p)
        # self.Expansion.update([ next(params) for i in range(self.Expansion.nPars) ])
        self.ExtraParams = [param for param in params]
        # self.nPar = len(p)

        # self.ModelObj.Correlate.fde_solve.reset_parameters(coef=self.Expansion)
        Robin_const = self.ExtraParams[0]

        self.sigma[:] = 0
        sigma_prev = np.zeros_like(self.sigma)
        for isample in range(self.nsamples):
            field = self.ModelObj.sample(Robin_const=Robin_const, adjoint=False)
            for i in range(3):
                self.sigma[i, :] += np.mean(field[:, :, :, i] ** 2, axis=(0, 1))
            err = np.linalg.norm(self.sigma / (isample + 1) - sigma_prev)
            if err < self.tol * np.linalg.norm(sigma_prev):
                break
            else:
                sigma_prev[:] = self.sigma / (isample + 1)
                print("{0:d}, {1:f}".format(isample, err))
        self.sigma /= self.nsamples

        return self.sigma

    def default_parameters(self):
        return 0


###################################################################################################
###################################################################################################


def Fourier_variance(WindModel, correlation_length=1.0):
    import scipy.fftpack as fft

    L, Nd, ndim = (
        WindModel.Correlate.L,
        WindModel.Correlate.Nd,
        WindModel.Correlate.ndim,
    )
    Nz = Nd[2]
    Nd[2] = 2 * Nd[2] - 1
    L[2] *= 2

    # Nd[0] = 2**8
    # Nd[1] = 2**8

    # Frequencies2 = [(2*pi/L[j])*(Nd[j]*fft.fftfreq(Nd[j])) for j in range(ndim)]
    # k = np.array(list(np.meshgrid(*Frequencies2, indexing='ij')))
    # kk = np.sum(k**2,axis=0)
    # k1 = k[0,...]

    # # tmp = 1 / (1/correlation_length**2+kk)**(17/6)
    # tmp = k1**2 / (1/correlation_length**2+kk)**(17/6)
    # tmp = fft.ifftn(tmp).real
    # a = tmp[0,0,0]
    # b = tmp[0,0,:Nz]

    # c = a-b
    # c /= c[-1]

    # Nd[0] = 2**8
    # Nd[1] = 2**8
    # Frequencies2 = [(2*pi/L[j])*(Nd[j]*fft.fftfreq(Nd[j])) for j in range(ndim)]
    # k = np.array(list(np.meshgrid(*Frequencies2, indexing='ij')))
    # kk = np.sum(k**2,axis=0)
    # k1 = k[0,...]

    # # k = (2*pi/L[2])*(Nd[2]*fft.fftfreq(Nd[2]))
    # tmp = k1**2 / (1/correlation_length**2+kk)**(11/6)
    # tmp = fft.ifftn(tmp).real
    # a = tmp[0,0,0]
    # b = tmp[0,0,:Nz]
    # c = b/a

    Nd[0] = 2**8
    Nd[1] = 2**5

    # Frequencies2 = [(2*pi/L[j])*(Nd[j]*fft.fftfreq(Nd[j])) for j in range(1,ndim)]
    # k = np.array(list(np.meshgrid(*Frequencies2, indexing='ij')))
    # kk = np.sum(k**2,axis=0)
    # k1 = k[0,...]
    # k3 = k[-1,...]
    # tmp = 1 / (1/correlation_length**2+kk)**(17/6)
    # tmp = k1**2 / (1/correlation_length**2+kk)**(17/6) * np.sqrt(k1**2)
    # tmp = (0.5*k1**2 + k3**2) / (1/correlation_length**2+kk)**(17/6) * np.sqrt(k1**2)
    # tmp = 0.5*k1**2 / (1/correlation_length**2+kk)**(17/6) * np.sqrt(k1**2)
    # tmp = np.mean(tmp, axis=0)
    # tmp = fft.ifft(tmp).real
    # a = tmp[0,0]
    # b = tmp[0,:Nz]
    # c = a-b
    # c = a+b
    # tmp = k3**2 / (1/correlation_length**2+kk)**(17/6) * np.sqrt(k1**2)
    # tmp = fft.ifftn(tmp).real
    # a = tmp[0,0]
    # b = tmp[0,:Nz]
    # c += a+b

    r = 1e4 * np.linspace(0, 1, Nd[1]) ** 6
    k3 = (2 * pi / L[2]) * (Nd[2] * fft.fftfreq(Nd[2]))
    r, k3 = np.array(list(np.meshgrid(*[r, k3], indexing="ij")))
    tmp = r**2 / (1 / correlation_length**2 + r**2 + k3**2) ** (17 / 6) * r
    tmp = (r[1:, 0] - r[:-1, 0]).T.dot(tmp[1:, :] + tmp[:-1, :]) / 2
    tmp = fft.ifft(tmp).real
    a = tmp[0]
    b = tmp[:Nz]
    c = a - b

    c /= c[-1]
    return c


###################################################################################################
###################################################################################################

if __name__ == "__main__":
    """TEST by method of manufactured solutions"""

    pass
