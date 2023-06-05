
from math import *
import numpy as np
import scipy.optimize
from collections.abc import Iterable
import torch
from pylab import *
from time import time, sleep
from multiprocessing import Process
from matplotlib.animation import FuncAnimation

from .GenericObjectiveFunction import GenericObjectiveFunction
from .Matrices import DerivativeStiffnessMatrixGenerator, DerivativeCovarianceMatrixGenerator



###################################################################################################
#   Rapid distortion one-point spectra
###################################################################################################

class USBL1pSpectraObjectiveFunction(GenericObjectiveFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ### Dimensions and grids
        self.nModes      = self.ModelObj.Correlate.fde_solve.nModes
        self.RA_coefs    = self.ModelObj.Correlate.fde_solve.c
        self.adjVecPot   = self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier
        self.grid_z      = self.ModelObj.Correlate.fde_solve.ode_solve.grid
        self.ndofs_z     = len(self.grid_z)
        self.ndofs_sys   = self.vdim*len(self.grid_z)

    ###------------------------------------------------
    ### Parameters
    ###------------------------------------------------

    @property
    def Common_parameters(self):
        return np.array([self.lengthscale, self.scalingfactor, self.sqrtRobin_const])

    @Common_parameters.setter
    def Common_parameters(self, params):
        self.lengthscale, self.scalingfactor, self.sqrtRobin_const = params

    def init_Common_parameters(self, **kwargs):
        self.lengthscale     = 0.59 #sqrt(self.ModelObj.Correlate.corrlen)
        self.scalingfactor   = 3.2
        self.sqrtRobin_const = 1.0 # np.infty

    ### Spacial metrics oeffiient
    def L_coef(self, z):
        return (self.lengthscale**2) ### constant law
        # return (self.lengthscale**2)*(z + 1.e-3) ### linear law

    #NOTE: NN parameters are defined in the parent class !


    #------------------------------------------------------------------------------------------
    # Computational part
    #------------------------------------------------------------------------------------------

    def initialize(self, **kwargs):
        self.init_Common_parameters(**kwargs)
        configEddyLifeTime = kwargs.get('configEddyLifeTime', {})
        self.EddyLifetime.Initialize(**configEddyLifeTime)

        ### Initialize Matrix generators
        domain_height = self.ModelObj.Correlate.fde_solve.domain_height
        dof  = self.ndofs_z
        coef = lambda z: z ### dummy !!!! update in update_parameters
        self.MatrixA = DerivativeStiffnessMatrixGenerator(dof, coef, self.EddyLifetime, domain_height=domain_height) ### TODO: creat object as empty as possible, since they updated only in update parameters.
        self.MatrixB = DerivativeCovarianceMatrixGenerator(dof, coef, self.EddyLifetime, domain_height=domain_height)

        ### Init EddyLifeTime
        configEddyLifeTime = kwargs.get('configEddyLifeTime', {})
        if configEddyLifeTime:
            self.EddyLifetime.Initialize(**configEddyLifeTime)

        ### Sizes
        self.nParams = len(self.All_parameters)

        ### Arrays
        self.f              = np.zeros([self.vdim, len(self.grid_k2), self.ndofs_sys], dtype=np.complex)
        self.f_tilde        = np.zeros([self.vdim, len(self.grid_k2), self.nModes, self.ndofs_sys], dtype=np.complex)
        self.lambda_tilde   = np.zeros([self.vdim, len(self.grid_k2), self.nModes, self.ndofs_sys], dtype=np.complex)
        self.B              = np.zeros([len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex)
        self.A              = np.zeros([len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex)
        self.DB             = np.zeros([self.nParams, len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex)
        self.DA             = np.zeros([self.nParams, len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex)
        self.SP             = np.zeros_like(self.DataValues)
        self.DSP            = np.zeros([self.nParams, self.nDataPoints, self.vdim, self.vdim])

        if self.verbose: print('Initialization complete.')

        ### TODO: add an option for Reynolds stress computation

    #------------------------------------------------------------------------------------------
    
    def update(self, theta=None, jac=False):
        if theta is not None: self.All_parameters = theta

        ### Update Matrices
        self.lengthscale, self.scalingfactor, self.sqrtRobin_const = self.Common_parameters
        self.ModelObj.Correlate.fde_solve.reset_ode(self.L_coef)
        self.MatrixA.update_coef(self.L_coef)  
        self.MatrixB.update_coef(self.L_coef)
        self.MatrixB.update_scale(self.scalingfactor)
        self.MatrixA.update_BC(self.sqrtRobin_const)
        self.MatrixA.update_stored_matrices() ### Update stored matrices (only actually required for Matrix A)
        self.MatrixA.update_eddy_lifetime(self.NN_parameters)
        self.MatrixB.update_eddy_lifetime(self.NN_parameters)


        ### Computation of the spetra and the objetive funtion

        J = 0
        if jac: DJ = np.zeros([len(self.All_parameters)])

        for l in range(self.nDataPoints):
            self.k1, self.z = self.DataPoints[l]

            for k, self.k2 in enumerate(self.grid_k2):
                self.tau  = self.EddyLifetime.eval(self.k1, self.k2)
                self.B[k] = self.MatrixB.assemble_matrix(self.k1, self.k2)
                self.A[k] = self.MatrixA.assemble_matrix(self.k1, self.k2)

                for i in range(self.vdim):
                    d_il              = self.compute_d(i)                            ### TODO: can be precomputed
                    self.f_tilde[i,k] = self.apply_Linv_split_modes(d_il, factor=self.RA_coefs)
                    self.f[i,k]       = np.sum(self.f_tilde[i,k], axis=0)
                    if jac: self.lambda_tilde[i,k] = self.apply_Linv_split_modes(self.B[k] @ self.f[i,k], factor=1)

                if jac:
                    DA = self.MatrixA.assemble_derivative(self.k1, self.k2)
                    DB = self.MatrixB.assemble_derivative(self.k1, self.k2)
                    for p in range(self.nParams):
                        self.DB[p,k] = DB[p]            
                        self.DA[p,k] = DA[p]             ### NOTE: slow fix

            for i in range(self.vdim):
                for j in range(self.vdim): ### TODO: accelerate due to symmetry of Reynolds stress
                    if i==j: ### NOTE: Only dioganal of Reynolds stress
                        self.SP[l,i,j] = self.compute_spectrum(i,j,l)
                        eps = self.SP[l,i,j] - self.DataValues[l,i,j]
                        # eps = self.compute_eps(i,j,l)
                        if True: #i==0: ### fit only specifi components
                            J += eps**2
                            if jac:
                                for p in range(self.nParams):
                                    Deps = self.compute_Deps(i,j,l,p)
                                    DJ[p] += eps*Deps
                                    self.DSP[p,l,i,j] = Deps

        self.J = 0.5*J
        if jac: self.DJ = DJ

        #==========================================
        ### data for the plot
        self.tau_model = self.EddyLifetime.eval(self.grid_k1, 0*self.grid_k1)
        #==========================================

        if self.verbose: print('Updated.')

    #------------------------------------------------------------------------------------------

    def compute_spectrum(self, i,j,l):
        spectrum = self.InnerProd(self.f[i], self.B, self.f[j])
        return self.k1*spectrum

    # def compute_eps(self, i,j,l):
    #     eps = self.InnerProd(self.f[i], self.B, self.f[j]) - self.DataValues[l,i,j]
    #     return eps

    def compute_Deps(self, i,j,l, p):
        Deps = self.InnerProd(self.f[i], self.DB[p], self.f[j])
        for n in range(self.nModes):
            Deps += -2*self.InnerProd(self.f_tilde[i,:,n], self.DA[p], self.lambda_tilde[j,:,n])
        return self.k1*Deps

    #------------------------------------------------------------------------------------------ 
           
    def compute_d(self, i):

        k1, k2, z = self.k1, self.k2, self.z

        assert(i >= 0)
        assert(i <= 2)
        assert(z >= self.grid_z.lower_bound)
        assert(z < self.grid_z.upper_bound)

        grid = self.grid_z[:]
        M = len(grid)
        d = np.zeros([3*M], dtype=np.complex)
        
        ind = np.sum(1*(grid<=z))-1
        h = grid[ind+1]-grid[ind]
        phi_z = 1 - (z-grid[ind])/h ### weight

        # (0, d/dz \phi, -i k2 \phi)
        if i == 0:                
            if phi_z < 1:
                d[ M + ind ] = -h
                d[ M + ind + 1 ] = h
                d[ 2*M + ind ] = - 1j * k2 * phi_z
                d[ 2*M + ind + 1 ] = - 1j * k2 * (1-phi_z)
            elif phi_z == 1:
                if z > 0:
                    d[ M + ind - 1 ] = -h/2.0
                    d[ M + ind + 1 ] = h/2.0
                else:
                    d[ M + ind ] = -h
                    d[ M + ind + 1 ] = h
                d[ 2*M + ind ] = - 1j * k2
        # (-d/dz \phi, 0, i k1 \phi)
        elif i == 1:
            if phi_z < 1:
                d[ ind ] = h
                d[ ind + 1 ] = -h
                d[ 2*M + ind ] = 1j * k1 * phi_z
                d[ 2*M + ind + 1 ] = 1j * k1 * (1-phi_z)
            elif phi_z == 1:
                if z > 0:
                    d[ ind - 1 ] = h/2.0
                    d[ ind + 1 ] = -h/2.0
                else:
                    d[ ind ] = h
                    d[ ind + 1 ] = -h
                d[ 2*M + ind ] = 1j * k1
        # (i k2 phi, -i k1 \phi, 0)
        else:
            d[ ind ] = 1j * k2 * phi_z
            d[ ind + 1 ] = 1j * k2 * (1-phi_z)
            d[ M + ind ] = - 1j * k1 * phi_z
            d[ M + ind + 1 ] = - 1j * k1 * (1-phi_z)
        
        return d

    #------------------------------------------------------------------------------------------
    # Helper methods
    #------------------------------------------------------------------------------------------

    def apply_Linv_split_modes(self, rhs, factor=1):
        M = self.ndofs_z
        y = np.zeros([self.nModes, self.vdim*M], dtype=np.complex)
        factor = factor if isinstance(factor, Iterable) else [factor]*self.nModes
        for n in range(self.nModes):
            for j in range(self.vdim):
                ind_j = j*M+np.arange(M)
                y[n,ind_j] = factor[n] * self.adjVecPot(rhs[ind_j], [self.k1], [self.k2], component=j, Robin_const=self.sqrtRobin_const**2, tau=self.tau, mode=n, noFactor=True)[0,0,:]
        return y


    def apply_Linv(self, rhs):
        M = self.ndofs_z
        y = np.zeros([self.vdim*M], dtype=np.complex)
        for j in range(self.vdim):
            ind_j = j*M+np.arange(M)
            y[ind_j] = self.adjVecPot(rhs[ind_j], [self.k1], [self.k2], component=j, Robin_const=self.sqrtRobin_const**2, tau=self.tau, noFactor=True)[0,0,:]
        return y

    def InnerProd(self, x, A, y):    ### TODO: accelerate in case of A matrix (block-diagonal)
        xAy = np.array([ y[k].conj() @ A[k] @ x[k] for k in range(len(self.grid_k2)) ])
        # imag_err = np.abs(xAy.imag).max()
        # if imag_err > 1.e-6:
        #     raise Exception('imag error = {0}'.format(imag_err))
        ### NOTE: inner product with DA is not supposed to pass this assertion, since we have (f, B*df) + (df, B*f),which we replace with 2*Re(f, B*df)
        xAy = xAy.real
        xAy = self.Quad_k2(xAy)
        return xAy


###################################################################################################
###################################################################################################

if __name__ == "__main__":

    '''TEST by method of manufactured solutions'''

    pass