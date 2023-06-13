import sys
sys.path.append("/Users/bk/Work/Papers/Collaborations/2020_inletgeneration/code/source/")
sys.path.append("/home/khristen/Projects/Brendan/2019_inletgeneration/code/source")

from math import *
from collections.abc import Iterable, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from fracturbulence.RandomFieldModule.utilities.ode_solve import FEM_coefficient_matrix_generator, Grid1D
from fracturbulence.RandomFieldModule.Calibration.EddyLifetime import *
import scipy.fftpack as fft


"""

    General class for computing a matrix and its derivatives for the specific frequency (k1,k2)

    Comments:
    Made inputs similar to ode_solve

"""
class DerivativeMatrixGenerator:

    def __init__(self, dof, L_coef, EddyLifetime, domain_height=1):
        self.grid = Grid1D(dof, upper_bound=domain_height)
        self.tau = EddyLifetime
        self.update_coef(L_coef)

    def update_eddy_lifetime(self,param_vec):
        self.tau.update_parameters(param_vec)

    def update_coef(self, L_coef):
        if isinstance(L_coef, Iterable):
            if len(L_coef)==1:
                self.L_coef = [L_coef[0]] * 3
            elif len(L_coef)==2:
                self.L_coef = [ L_coef[0], L_coef[0], L_coef[1] ]
            else:
                self.L_coef = L_coef
        else:
            self.L_coef = [L_coef] * 3
        self.lengthscale = sqrt(self.L_coef[0](1.0))

    def update_stored_matrices(self):
        return 0

    def assemble_matrix(self, **kwargs):
        return 0

    def assemble_derivative(self, k1, k2):
        return 0

"""

    Assembles A and dA/dparams for the specific frequency (k1,k2)

"""
class DerivativeStiffnessMatrixGenerator(DerivativeMatrixGenerator):

    def quadrature_points(self):
        qp = np.zeros([2, len(self.grid)-1])
        qp[0,:] = (self.grid[1:] - self.grid.h/2) - 1/sqrt(3) * self.grid.h/2
        qp[1,:] = (self.grid[1:] - self.grid.h/2) + 1/sqrt(3) * self.grid.h/2
        return qp

    def __init__(self, dof, L_coef, EddyLifetime, domain_height=1.0, sqrtRobin_const=0.0):
        super().__init__(dof, L_coef, EddyLifetime, domain_height)
        self.qp = self.quadrature_points()
        self.update_BC(sqrtRobin_const)
        self.update_stored_matrices()

    def update_BC(self, sqrtRobin_const):
        self.sqrtRobin_const = sqrtRobin_const

    def update_stored_matrices(self):
        self.Mass0 = self.mass_matrix(self.func_const1())
        self.Mass1 = self.mass_matrix(self.L_coef[0])
        self.Mass2 = self.mass_matrix(self.L_coef[1])
        self.Mass3 = self.mass_matrix(self.L_coef[2])
        self.Cross = self.crossterm_diffusion_matrix(self.L_coef[2])
        self.Diff  = self.diffusion_matrix(self.L_coef[2])

        m = len(self.grid)
        self.dAdscalingfactor = np.zeros((3*m,3*m), dtype=np.complex128)
        self.dAdsqrtRobin_const = np.zeros((3*m,3*m), dtype=np.complex128)
        self.dAdsqrtRobin_const[2*m,2*m] = 2.0 * self.sqrtRobin_const

    @staticmethod
    def func_const1():
        return (lambda z : 1)

    def assemble_matrix(self, k1, k2, d=0):
        tau = self.tau.eval(k1,k2)
        M0 = self.Mass0
        M1 = self.Mass1
        M2 = self.Mass2
        M3 = self.Mass3
        C = self.Cross
        D = self.Diff
        # A11 = (d+1)*M + (k1**2)*M1 + (k2**2)*M2 + ((tau*k1)**2)*M3 + (tau*k1)*C + D
        # return A11

        tmp = (d+1)*M0 + (k1**2)*M1 + (k2**2)*M2 + ((tau*k1)**2)*M3 + (tau*k1)*C + D
        m = len(self.grid)
        A = np.zeros((3*m,3*m), dtype=np.complex128)
        A11 = np.diag(tmp[0,1:],k=1) + np.diag(tmp[1,:],k=0) + np.diag(tmp[2,:-1],k=-1)
        A[:m,:m] = A11
        A[m:2*m,m:2*m] = A11
        A[2*m:3*m,2*m:3*m] = A11
        A[2*m,2*m] += self.sqrtRobin_const**2
        return A

    def assemble_derivative(self, k1, k2):
        tau = self.tau.eval(k1,k2)
        dtau = self.tau.eval_deriv(k1,k2)
        M1 = self.Mass1
        M2 = self.Mass2
        M3 = self.Mass3
        C = self.Cross
        D = self.Diff

        # dA11dtau = 2*tau*(k1**2)*M3 + k1*C
        # return [dA11dtau * dtau_i for dtau_i in dtau]

        tmp1 = 2.0*tau*(k1**2)*M3 + k1*C
        m = len(self.grid)
        dA11dtau = np.diag(tmp1[0,1:],k=1) + np.diag(tmp1[1,:],k=0) + np.diag(tmp1[2,:-1],k=-1)

        dAdtau = np.zeros((3*m,3*m), dtype=np.complex128)
        dAdtau[:m,:m] = dA11dtau
        dAdtau[m:2*m,m:2*m] = dA11dtau
        dAdtau[2*m:3*m,2*m:3*m] = dA11dtau

        tmp2 = (k1**2)*M1 + (k2**2)*M2 + ((tau*k1)**2)*M3 + (tau*k1)*C + D
        tmp2 = 2.0 * tmp2 / self.lengthscale # trick assuming L^2(z) = lengthscale^2 * z^beta
        dA11dlengthscale = np.diag(tmp2[0,1:],k=1) + np.diag(tmp2[1,:],k=0) + np.diag(tmp2[2,:-1],k=-1)

        dAdlengthscale = np.zeros((3*m,3*m), dtype=np.complex128)
        dAdlengthscale[:m,:m] = dA11dlengthscale
        dAdlengthscale[m:2*m,m:2*m] = dA11dlengthscale
        dAdlengthscale[2*m:3*m,2*m:3*m] = dA11dlengthscale

        return [dAdlengthscale] + [self.dAdscalingfactor] + [self.dAdsqrtRobin_const] \
             + [dAdtau * dtau_i for dtau_i in dtau]

    def mass_matrix(self, reaction_function):
        '''creates the mass matrix in matrix diagonal ordered form'''
        M = np.zeros((3,len(self.grid)))
        fun_qp = reaction_function(self.qp)
        if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
        w = 1/2
        x1 = (1-1/np.sqrt(3))/2
        x2 = (1+1/np.sqrt(3))/2
        coeff_element_0 = (fun_qp[0,:]*(x1**2) + fun_qp[1,:]*(x2**2)) * self.grid.h * w
        coeff_element_1 = (fun_qp[0,:]*(x1*(1-x1)) + fun_qp[1,:]*(x2*(1-x2))) * self.grid.h * w
        M[0,1:]  = coeff_element_1
        M[1,:-1] = coeff_element_0
        M[1,1:] += coeff_element_0
        M[2,:-1] = M[0,1:]
        return M

    def crossterm_diffusion_matrix(self, crossterm_diffusion_function):
        '''creates the cross-term contributions to the diffusion matrix in matrix diagonal ordered form
           NOTE that this contribution is ONLY active when t > 0 '''
        D = np.zeros((3,len(self.grid)))
        fun_qp = crossterm_diffusion_function(self.qp)
        if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
        w = 1/2
        x1 = (1-1/np.sqrt(3))/2
        x2 = (1+1/np.sqrt(3))/2
        coeff_element = 2*(fun_qp[0,:]*x1 + fun_qp[1,:]*x2) * w
        D[0,1:]  = -coeff_element
        D[2,:-1] = -D[0,1:]
        return 1j*D
    
    def diffusion_matrix(self, diffusion_function):
        '''creates the diffusion matrix in matrix diagonal ordered form'''
        D = np.zeros((3,len(self.grid)))
        fun_qp = diffusion_function(self.qp)
        if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
        w = 1/2
        coeff_element = fun_qp.sum(axis=0)*w/self.grid.h
        D[0,1:]  = -coeff_element
        D[1,:-1] = coeff_element
        D[1,1:] += coeff_element
        D[2,:-1] = D[0,1:]
        return D

"""

    Assembles B and dB/dparams for the specific frequency (k1,k2)

"""
class DerivativeCovarianceMatrixGenerator(DerivativeMatrixGenerator):

    def __init__(self, dof, L_coef, EddyLifetime, domain_height=1, scalingfactor=1.0):
        super().__init__(dof, L_coef, EddyLifetime, domain_height)
        self.compute_k3()
        self.compute_fourier_basis()
        self.update_scale(scalingfactor)
        self.update_stored_matrices()

    def update_scale(self, scalingfactor):
        self.scalingfactor = scalingfactor

    def compute_k3(self):
        N = len(self.grid)
        L = self.grid.upper_bound - self.grid.lower_bound
        self.k3 = (2*pi/L)*(N*fft.fftfreq(N))

    def compute_fourier_basis(self):
        assert(isinstance(self.grid.h, float))
        h = self.grid.h
        M = len(self.grid)
        phi_hat = np.zeros((M,M),dtype=np.complex128)
        for l in range(M):
            phi_hat[l,:] = np.sinc(h*self.k3/(2*np.pi))**2 * np.exp(-1j * h * l * self.k3) * h / np.sqrt(2*np.pi)
        self.phi_hat = phi_hat
    
    def update_stored_matrices(self):
        m = len(self.grid)
        self.dBdsqrtRobin_const = np.zeros((3*m,3*m), dtype=np.complex128)

    def assemble_matrix(self, k1, k2, multiply_by_detTheta=True):

        assert(isinstance(k1, float))
        assert(isinstance(k2, float))
        
        tau = self.tau.eval(k1,k2)
        if multiply_by_detTheta:
            alpha = 17/12
            detTheta = self.L_coef[0](self.grid[:])*self.L_coef[1](self.grid[:])*self.L_coef[2](self.grid[:])
            detTheta = np.power(detTheta,alpha/3)

        k3 = self.k3
        k30  = k3 + tau*k1

        kk = k1**2 + k2**2 + k3**2
        kk0 = k1**2 + k2**2 + k30**2

        s = k1**2 + k2**2
        C1  =  tau * k1**2 * (kk0 - 2 * k30**2 + tau * k1 * k30) / (kk * s)
        tmp =  tau * k1 * np.sqrt(s) / (kk0 - k30 * k1 * tau)
        C2  =  k2 * kk0 / s**(3/2) * np.arctan (tmp)

        zeta1_by_zeta3 =  (C1 - k2/k1 * C2) * (kk/kk0)
        zeta2_by_zeta3 =  (k2/k1 * C1 + C2) * (kk/kk0)
        one_by_zeta3   =  kk/kk0

        zeta1_by_zeta3[np.isnan(zeta1_by_zeta3)] = 0
        zeta2_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0
        one_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0

        M = len(self.grid[:])
        G = np.zeros((3,3,M))
        G[0,0,:] = 1;        G[0,1,:] = 0;        G[0,2,:] = -zeta1_by_zeta3
        G[1,0,:] = 0;        G[1,1,:] = 1;        G[1,2,:] = -zeta2_by_zeta3
        G[2,0,:] = G[0,2,:]; G[2,1,:] = G[1,2,:]; G[2,2,:] = zeta1_by_zeta3**2 + zeta2_by_zeta3**2 + one_by_zeta3**2

        # NOTE: Can be optimized (B11 and B22 are mass matrices, B12 is zero)
        B = np.zeros((3*M,3*M),dtype=np.complex128)
        phi_hat = self.phi_hat
        for i in range(3):
            for j in range(3):
                for l in range(M):
                    for m in range(M):
                        B[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * G[i,j,:] * phi_hat[m,:].conj())
                if multiply_by_detTheta:
                    for l in range(M):
                        B[i*M + l,j*M:(j+1)*M] *= detTheta
                    for m in range(M):
                        B[i*M:(i+1)*M,j*M + m] *= detTheta

        return B * pow(self.scalingfactor,2)

    '''create list of FEM matrices in so-called "matrix diagonal ordered form", indexed by the NN parameters '''
    def assemble_derivative(self, k1, k2, multiply_by_detTheta=True):

        assert(isinstance(k1, float))
        assert(isinstance(k2, float))

        tau = self.tau.eval(k1,k2)
        if multiply_by_detTheta:
            alpha = 17/12
            detTheta = self.L_coef[0](self.grid[:])*self.L_coef[1](self.grid[:])*self.L_coef[2](self.grid[:])
            detTheta = np.power(detTheta,alpha/3)

        k3 = self.k3
        k30  = k3 + tau*k1

        kk = k1**2 + k2**2 + k3**2
        kk0 = k1**2 + k2**2 + k30**2
        s = k1**2 + k2**2
        
        # C1 and derivative
        C1  =  tau * k1**2 * (kk0 - 2 * k30**2 + tau * k1 * k30) / (kk * s)
        dC1dtau = k1**2 * (kk0 - 2 * k30**2 + tau**2 * k1**2) / (kk * s)
        # dC1dtau = k1**2 * (kk0 - 2 * k30**2 + 2 * tau * k1 * k30) / (kk * s)


        a = k1 * np.sqrt(s)
        b = kk
        c = k3 * k1
        tmp1 = np.arctan( a * tau  / (b + c * tau) )
        tmp2 = a * b / ( (a * tau)**2  + (b + c * tau)**2)
        # C2 and derivative
        C2  =  k2 * kk0 / s**(3/2) * tmp1
        dC2dtau = (k2 / s**(3/2)) * ( kk0 * tmp2 + 2.0 * k1 * k30 * tmp1 )

        # zetas and derivatives
        zeta1 = C1 - k2/k1 * C2
        zeta2 = k2/k1 * C1 + C2
        zeta3 = kk0/kk
        dzeta1dtau = dC1dtau - k2/k1 * dC2dtau
        dzeta2dtau = k2/k1 * dC1dtau + dC2dtau
        dzeta3dtau = 2.0 * k1 * k30 / kk

        # non-zero components of dGdtau
        dG13dtau = - dzeta1dtau / zeta3 + zeta1 * dzeta3dtau / zeta3**2 
        dG23dtau = - dzeta2dtau / zeta3 + zeta2 * dzeta3dtau / zeta3**2 
        dG33dtau = 2.0 * ( zeta1 * dzeta1dtau + zeta2 * dzeta2dtau ) / zeta3**2 \
            - 2.0 * dzeta3dtau * ( zeta1**2 + zeta2**2 + 1.0 ) / zeta3**3
        
        dG13dtau[np.isnan(dG13dtau)] = 0
        dG23dtau[np.isnan(dG23dtau)] = 0
        dG33dtau[np.isnan(dG33dtau)] = 0

        # dBdtau
        M = len(self.grid[:])
        dBdtau = np.zeros((3*M,3*M),dtype=np.complex128)
        phi_hat = self.phi_hat

        i=2; j=0
        for l in range(M):
            for m in range(M):
                dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG13dtau * phi_hat[m,:].conj())
        
        i=2; j=1
        for l in range(M):
            for m in range(M):
                dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG23dtau * phi_hat[m,:].conj())

        i=2; j=2
        for l in range(M):
            for m in range(l,M):
                dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG33dtau * phi_hat[m,:].conj())

        dBdtau = dBdtau + dBdtau.T.conj() - np.diag(dBdtau.diagonal().conj())

        for i in range(3):
            for j in range(3):
                if multiply_by_detTheta:
                    for l in range(M):
                        dBdtau[i*M + l,j*M:(j+1)*M] *= detTheta
                    for m in range(M):
                        dBdtau[i*M:(i+1)*M,j*M + m] *= detTheta
        dBdtau *= pow(self.scalingfactor,2)

        B = self.assemble_matrix(k1, k2, multiply_by_detTheta=True)
        dBdscalingfactor = 2.0 * B / self.scalingfactor
        dBdlengthscale = 4.0 * alpha * B / self.lengthscale

        # dBdlengthscale = self.assemble_matrix(k1, k2, multiply_by_detTheta=False)
        # for i in range(3):
        #     for j in range(3):
        #         for l in range(M):
        #             dBdlengthscale[i*M:(i+1)*M,j*M + m] *= 2.0 * alpha * detTheta / self.lengthscale
        #         for m in range(M):
        #             dBdlengthscale[i*M:(i+1)*M,j*M + m] *= detTheta
        # dBdlengthscale = dBdlengthscale + dBdlengthscale.T.conj() - np.diag(dBdlengthscale.diagonal().conj())

        dtau = self.tau.eval_deriv(k1,k2)
        return [dBdlengthscale] + [dBdscalingfactor] + [self.dBdsqrtRobin_const] \
             + [dBdtau * dtau_i for dtau_i in dtau]
        # return [dBdtau * dtau_i for dtau_i in dtau]



############################################################################
############################################################################

if __name__ == "__main__":

    frequencies = np.array([[1.0, -1.0] ,[-2.0, 4.0]])
    tau0 = 0.0
    hidden_layer_size=2
    noise_magnitude = 1.0
    EddyLifetimeGrid = EddyLifetimeGrid(frequencies, tau0=tau0, hidden_layer_size=hidden_layer_size, noise_magnitude=noise_magnitude)
    print(EddyLifetimeGrid.parameters)
    print(EddyLifetimeGrid.values)
    print(EddyLifetimeGrid.derivatives)

    EddyLifetime = EddyLifetime(tau0=1, hidden_layer_size=5)
    L1 = lambda z : z
    L2 = lambda z : z**2
    L3 = lambda z : z**3
    coef = kappa = [ np.vectorize(lambda z: L1(z)**2), np.vectorize(lambda z: L2(z)**2), np.vectorize(lambda z: L3(z)**2) ]

    domain_height = 0.9
    dof = 2**6
    DerivativeStiffnessMatrixGenerator = DerivativeStiffnessMatrixGenerator(dof, coef, EddyLifetime, domain_height=domain_height)
    DerivativeCovarianceMatrixGenerator = DerivativeCovarianceMatrixGenerator(dof, coef, EddyLifetime, domain_height=domain_height)
    
    d = 0.0
    k1 = 1.0
    k2 = np.pi
    A = DerivativeStiffnessMatrixGenerator.assemble_matrix(d, k1, k2)
    DA = DerivativeStiffnessMatrixGenerator.assemble_derivative(k1, k2)
    B = DerivativeCovarianceMatrixGenerator.assemble_matrix(k1, k2)
    DB = DerivativeCovarianceMatrixGenerator.assemble_derivative(k1, k2)

    print(len(DerivativeStiffnessMatrixGenerator.tau.parameters))
    print(np.array(DA).shape)
    print(np.array(DB).shape)



