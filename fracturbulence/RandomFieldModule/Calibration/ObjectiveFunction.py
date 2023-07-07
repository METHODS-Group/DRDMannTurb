from collections.abc import Iterable
from math import *
from multiprocessing import Process
from time import sleep, time

import numpy as np
import scipy.optimize
import torch
from matplotlib.animation import FuncAnimation
from pylab import *

from fracturbulence.RandomFieldModule.PowerSpectra import (MannEddyLifetime,
                                                           StdEddyLifetime)

from .EddyLifetime import EddyLifetime
from .Matrices import (DerivativeCovarianceMatrixGenerator,
                       DerivativeStiffnessMatrixGenerator)

# sys.path.append("/Users/bk/Work/Papers/Collaborations/2020_inletgeneration/code/source/")
# sys.path.append("/home/bkeith/Work/Papers/2020_inletgeneration/code/source/")
# sys.path.append("/home/khristen/Projects/Brendan/2019_inletgeneration/code/source")




###################################################################################################
#  Generic Objective function
###################################################################################################


class ObjectiveFunction:
    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)
        self.tol = kwargs.get("tol", 1.0e-3)

        self.dev = kwargs.get("dev", False)  ### development mode (testing)
        if self.dev:
            self.verbose = True

    def __call__(self, theta):
        return 0

    def Jacobian(self):
        return None

    def initialize(self, **kwargs):
        pass

    def finalize(self, **kwargs):
        pass

    def default_parameters(
        self,
    ):
        return 0


###################################################################################################
#   Rapid distortion one-point spectra
###################################################################################################


class UniformShearOnePointSpectraObjectiveFunction(ObjectiveFunction):
    def __init__(self, ModelObj, Data, **kwargs):
        super().__init__(**kwargs)
        self.ModelObj = ModelObj
        self.DataPoints, self.DataValues = Data

        ### Parameters
        tau0 = kwargs.get("tau0", 0.0)
        hidden_layer_size = kwargs.get("hidden_layer_size", 5)
        self.EddyLifetime = EddyLifetime(tau0=tau0, hidden_layer_size=hidden_layer_size)
        self.nNNParams = self.EddyLifetime.parameters.size
        self.nExtraParams = kwargs.get(
            "nExtraParameters", 3
        )  ### e.g., corr_len, fscaling factor, Robin const
        self.nParams = self.nNNParams + self.nExtraParams

        ### Dimensions and grids
        self.vdim = self.ModelObj.vdim
        self.nDataPoints = len(self.DataPoints)
        self.nModes = self.ModelObj.Correlate.fde_solve.nModes
        self.RA_coefs = self.ModelObj.Correlate.fde_solve.c
        self.adjVecPot = (
            self.ModelObj.Correlate.compute_vector_potential_nonuniform_Fourier
        )
        self.grid_z = self.ModelObj.Correlate.fde_solve.ode_solve.grid
        self.grid_k2 = (2 * pi) * np.r_[
            -np.logspace(-2, 4, 20)[::-1], 0, np.logspace(-2, 4, 20)
        ]
        self.dk2 = self.grid_k2[1:] - self.grid_k2[:-1]
        self.ndofs_z = len(self.grid_z)
        self.ndofs_sys = self.vdim * len(self.grid_z)

        ### Matrix generators
        domain_height = self.ModelObj.Correlate.fde_solve.domain_height
        dof = self.ndofs_z
        coef = lambda z: z  ### dummy !!!! update in update_parameters
        self.MatrixA = DerivativeStiffnessMatrixGenerator(
            dof, coef, self.EddyLifetime, domain_height=domain_height
        )  ### TODO: creat object as empty as possible, since they updated only in update parameters.
        self.MatrixB = DerivativeCovarianceMatrixGenerator(
            dof, coef, self.EddyLifetime, domain_height=domain_height
        )

    # --------------------------------------------------------------------------
    #   Properties
    # --------------------------------------------------------------------------

    @property
    def lengthscale(self):
        return self.__lengthscale

    def L_coef(self, z):
        return self.lengthscale**2  ### constant law
        # return (self.lengthscale**2)*(z + 1.e-3) ### linear law

    @lengthscale.setter
    def lengthscale(self, lengthscale):
        self.__lengthscale = lengthscale
        self.ModelObj.Correlate.fde_solve.reset_ode(self.L_coef)

        ### TODO: BK: add update L to DerivativeStiffnessMatrixGenerator
        self.MatrixA.update_coef(self.L_coef)
        self.MatrixB.update_coef(self.L_coef)
        # self.MatrixA = DerivativeStiffnessMatrixGenerator(  len(self.ModelObj.Correlate.fde_solve.ode_solve.grid),
        #                                                     self.L_coef,
        #                                                     self.EddyLifetime,
        #                                                     domain_height=self.ModelObj.Correlate.fde_solve.domain_height,
        #                                                     sqrtkappa=0)

    @property
    def scalingfactor(self):
        return self.__scalingfactor

    @scalingfactor.setter
    def scalingfactor(self, scalingfactor):
        self.__scalingfactor = scalingfactor
        # self.ModelObj.Correlate.factor = (scalingfactor)**2
        self.MatrixB.update_scale(scalingfactor)
        # self.ModelObj.Correlate.factor = 1 ### scaling DataValues and NOT inside Correlate. TODO: change it puttting scaling factor into the model spectra

    @property
    def sqrtRobin_const(self):
        return self.__sqrtRobin_const

    @sqrtRobin_const.setter
    def sqrtRobin_const(self, RC):
        self.__sqrtRobin_const = RC
        self.MatrixA.update_BC(self.sqrtRobin_const)

    # ------------------------------------------------------------------------------------------

    def default_parameters(self):
        self.lengthscale = 1.1  # sqrt(self.ModelObj.Correlate.corrlen)
        self.scalingfactor = 4.2
        self.sqrtRobin_const = 1.0  # np.infty

        # extra_params = [3,  2.86892107,  0.46569357] #[self.lengthscale, self.scalingfactor, self.sqrtRobin_const]
        extra_params = [self.lengthscale, self.scalingfactor, self.sqrtRobin_const]
        NN_params = list(
            1.0e-4 * np.random.randn(self.nNNParams)
        )  # NOTE: Starting with params = 0, makes derivative of NN wrt params = [0,0,...,0,1]
        # NN_params    = [0]*self.nNNParams
        # NN_params    = [  12.60975289, -19.53594908,  -0.16514587,  32.90647554,   2.77274571,  -1.04525185,  -1.04522465]
        params = extra_params[: self.nExtraParams] + NN_params
        return params

    def update_parameters(self, theta):
        assert len(theta) == self.nParams
        NN_params = theta[self.nExtraParams :]
        extra_params = iter(theta[: self.nExtraParams])

        ### Lengthscales
        try:
            self.lengthscale = next(extra_params)
        except:
            pass

        ### Scaling factor
        try:
            self.scalingfactor = next(extra_params)
        except:
            pass

        ### RC
        try:
            self.sqrtRobin_const = next(extra_params)
            # if not np.isinf(self.sqrtRobin_const): self.sqrtRobin_const *= self.lengthscale
        except:
            pass

        ### Update stored matrices (only actually required for Matrix A)
        self.MatrixA.update_stored_matrices()

        ### Eddy lifetime
        self.EddyLifetime.update_parameters(NN_params)
        self.MatrixA.update_eddy_lifetime(NN_params)
        self.MatrixB.update_eddy_lifetime(NN_params)

        ### Print pytorch NN parameters (just to check)
        # print(list(self.EddyLifetime.NN.parameters()))

    def get_parameters(self):
        NN_params = self.EddyLifetime.parameters.tolist()
        extra_params = [self.lengthscale, self.scalingfactor, self.sqrtRobin_const]
        params = extra_params[: self.nExtraParams] + NN_params
        return params

    # ------------------------------------------------------------------------------------------

    def __call__(self, theta):
        if self.dev:  ### dev mode
            p = 5
            l = 2
            i = 0

            self.update(theta)
            if self.verbose:
                print("Function call: J = {0}".format(self.J))
                print("theta =\n", theta)

            J0 = self.J.copy()
            SP0 = self.SP[l, i, i].copy()
            B0 = self.B.copy()
            A0 = self.A.copy()

            h = 1.0e-4
            theta[p] += h
            self.update(theta, jac=True)
            if self.verbose:
                print("Function call: J = {0}".format(self.J))
                print("theta =\n", theta)

            J1 = self.J.copy()
            SP1 = self.SP[l, i, i].copy()
            B1 = self.B.copy()
            A1 = self.A.copy()

            print("FD J:     ", (J1 - J0) / h)
            print(
                "error DSP:     ", np.abs((SP1 - SP0) / h - self.DSP[p, l, i, i]).max()
            )
            print("error DB:     ", np.abs((B1 - B0) / h - self.DB[p]).max())
            print("error DA:     ", np.abs((A1 - A0) / h - self.DA[p]).max())
            print("norm A:     ", np.abs(A1).max())
            print("norm DA:     ", np.abs(self.DA[p]).max())
            print("norm DB:     ", np.abs(self.DB[p]).max())
            print(
                "norm DSP:     ",
                np.abs((SP1 - SP0) / h).max(),
                np.abs(self.DSP[p, l, i, i]).max(),
            )

            return self.J

        else:  ### high user mode
            self.update(theta)
            if self.verbose:
                print("Function call:")
                print("   J = {0}".format(self.J))
                print(
                    ("   theta = [" + ", ".join(["{}"] * len(theta)) + "]\n").format(
                        *theta
                    )
                )
            self.plot(dynamic=True)
            return self.J

    # ------------------------------------------------------------------------------------------

    def Jacobian(self, theta):
        self.update(theta, jac=True)
        if self.verbose:
            print("Jacobian call:")
            print("   ||DJ|| = {0}".format(np.linalg.norm(self.DJ)))
            print(
                ("   DJ = [" + ", ".join(["{}"] * len(self.DJ)) + "]\n").format(
                    *self.DJ
                )
            )
        return self.DJ

    # ------------------------------------------------------------------------------------------
    # Computational part
    # ------------------------------------------------------------------------------------------

    def initialize(self, **kwargs):
        self.update_parameters(self.default_parameters())

        ### Init EddyLifeTime
        configEddyLifeTime = kwargs.get("configEddyLifeTime", {})
        if configEddyLifeTime:
            self.EddyLifetime.Initialize(**configEddyLifeTime)
            self.MatrixA.update_eddy_lifetime(self.EddyLifetime.parameters)
            self.MatrixB.update_eddy_lifetime(self.EddyLifetime.parameters)

        ### Arrays
        self.f = np.zeros(
            [self.vdim, len(self.grid_k2), self.ndofs_sys], dtype=np.complex
        )
        self.f_tilde = np.zeros(
            [self.vdim, len(self.grid_k2), self.nModes, self.ndofs_sys],
            dtype=np.complex,
        )
        self.lambda_tilde = np.zeros(
            [self.vdim, len(self.grid_k2), self.nModes, self.ndofs_sys],
            dtype=np.complex,
        )
        self.B = np.zeros(
            [len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex
        )
        self.A = np.zeros(
            [len(self.grid_k2), self.ndofs_sys, self.ndofs_sys], dtype=np.complex
        )
        self.DB = np.zeros(
            [self.nParams, len(self.grid_k2), self.ndofs_sys, self.ndofs_sys],
            dtype=np.complex,
        )
        self.DA = np.zeros(
            [self.nParams, len(self.grid_k2), self.ndofs_sys, self.ndofs_sys],
            dtype=np.complex,
        )
        self.SP = np.zeros_like(self.DataValues)
        self.DSP = np.zeros([self.nParams, self.nDataPoints, self.vdim, self.vdim])

        if self.verbose:
            print("Initialization complete.")

        ### TODO: add an option for Reynolds stress computation

    # ------------------------------------------------------------------------------------------

    def finalize(self, **kwargs):
        print("Calibration finished.")
        self.plot(dynamic=False)

    # ------------------------------------------------------------------------------------------

    def update(self, theta=None, jac=False):
        if theta is not None:
            self.update_parameters(theta)

        J = 0
        if jac:
            DJ = np.zeros([len(theta)])

        for l in range(self.nDataPoints):
            self.k1, self.z = self.DataPoints[l]

            for k, self.k2 in enumerate(self.grid_k2):
                self.tau = self.EddyLifetime.eval(self.k1, self.k2)
                self.B[k] = self.MatrixB.assemble_matrix(self.k1, self.k2)
                self.A[k] = self.MatrixA.assemble_matrix(self.k1, self.k2)

                for i in range(self.vdim):
                    d_il = self.compute_d(i)  ### TODO: can be precomputed
                    self.f_tilde[i, k] = self.apply_Linv_split_modes(
                        d_il, factor=self.RA_coefs
                    )
                    self.f[i, k] = np.sum(self.f_tilde[i, k], axis=0)
                    if jac:
                        self.lambda_tilde[i, k] = self.apply_Linv_split_modes(
                            self.B[k] @ self.f[i, k], factor=1
                        )

                if jac:
                    DA = self.MatrixA.assemble_derivative(self.k1, self.k2)
                    DB = self.MatrixB.assemble_derivative(self.k1, self.k2)
                    for p in range(self.nParams):
                        self.DB[p, k] = DB[p]
                        self.DA[p, k] = DA[p]  ### NOTE: slow fix

            for i in range(self.vdim):
                for j in range(
                    self.vdim
                ):  ### TODO: accelerate due to symmetry of Reynolds stress
                    if i == j:  ### NOTE: Only dioganal of Reynolds stress
                        self.SP[l, i, j] = self.compute_spectrum(i, j, l)
                        eps = (
                            self.SP[l, i, j] - self.DataValues[l, i, j]
                        ) / self.DataValues[l, i, j]
                        # eps = self.compute_eps(i,j,l)
                        if True:  # i==0: ### fit only specifi components
                            J += eps**2
                            if jac:
                                for p in range(self.nParams):
                                    Deps = (
                                        self.compute_Deps(i, j, l, p)
                                        / self.DataValues[l, i, j]
                                    )
                                    DJ[p] += eps * Deps
                                    self.DSP[p, l, i, j] = Deps

        self.J = 0.5 * J
        if jac:
            self.DJ = DJ

        if self.verbose:
            print("Updated.")

    # ------------------------------------------------------------------------------------------

    def compute_spectrum(self, i, j, l):
        spectrum = self.InnerProd(self.f[i], self.B, self.f[j])
        return self.k1 * spectrum

    # def compute_eps(self, i,j,l):
    #     eps = self.InnerProd(self.f[i], self.B, self.f[j]) - self.DataValues[l,i,j]
    #     return eps

    def compute_Deps(self, i, j, l, p):
        Deps = self.InnerProd(self.f[i], self.DB[p], self.f[j])
        for n in range(self.nModes):
            Deps += -2 * self.InnerProd(
                self.f_tilde[i, :, n], self.DA[p], self.lambda_tilde[j, :, n]
            )
        return self.k1 * Deps

    # ------------------------------------------------------------------------------------------

    def compute_d(self, i):
        k1, k2, z = self.k1, self.k2, self.z

        assert i >= 0
        assert i <= 2
        assert z >= self.grid_z.lower_bound
        assert z < self.grid_z.upper_bound

        grid = self.grid_z[:]
        M = len(grid)
        d = np.zeros([3 * M], dtype=np.complex)

        ind = np.sum(1 * (grid <= z)) - 1
        h = grid[ind + 1] - grid[ind]
        phi_z = 1 - (z - grid[ind]) / h  ### weight

        # (0, d/dz \phi, -i k2 \phi)
        if i == 0:
            if phi_z < 1:
                d[M + ind] = -h
                d[M + ind + 1] = h
                d[2 * M + ind] = -1j * k2 * phi_z
                d[2 * M + ind + 1] = -1j * k2 * (1 - phi_z)
            elif phi_z == 1:
                if z > 0:
                    d[M + ind - 1] = -h / 2.0
                    d[M + ind + 1] = h / 2.0
                else:
                    d[M + ind] = -h
                    d[M + ind + 1] = h
                d[2 * M + ind] = -1j * k2
        # (-d/dz \phi, 0, i k1 \phi)
        elif i == 1:
            if phi_z < 1:
                d[ind] = h
                d[ind + 1] = -h
                d[2 * M + ind] = 1j * k1 * phi_z
                d[2 * M + ind + 1] = 1j * k1 * (1 - phi_z)
            elif phi_z == 1:
                if z > 0:
                    d[ind - 1] = h / 2.0
                    d[ind + 1] = -h / 2.0
                else:
                    d[ind] = h
                    d[ind + 1] = -h
                d[2 * M + ind] = 1j * k1
        # (i k2 phi, -i k1 \phi, 0)
        else:
            d[ind] = 1j * k2 * phi_z
            d[ind + 1] = 1j * k2 * (1 - phi_z)
            d[M + ind] = -1j * k1 * phi_z
            d[M + ind + 1] = -1j * k1 * (1 - phi_z)

        return d

    # ------------------------------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------------------------------

    def apply_Linv_split_modes(self, rhs, factor=1):
        M = self.ndofs_z
        y = np.zeros([self.nModes, self.vdim * M], dtype=np.complex)
        factor = factor if isinstance(factor, Iterable) else [factor] * self.nModes
        for n in range(self.nModes):
            for j in range(self.vdim):
                ind_j = j * M + np.arange(M)
                y[n, ind_j] = (
                    factor[n]
                    * self.adjVecPot(
                        rhs[ind_j],
                        [self.k1],
                        [self.k2],
                        component=j,
                        Robin_const=self.sqrtRobin_const**2,
                        tau=self.tau,
                        mode=n,
                        noFactor=True,
                    )[0, 0, :]
                )
        return y

    def apply_Linv(self, rhs):
        M = self.ndofs_z
        y = np.zeros([self.vdim * M], dtype=np.complex)
        for j in range(self.vdim):
            ind_j = j * M + np.arange(M)
            y[ind_j] = self.adjVecPot(
                rhs[ind_j],
                [self.k1],
                [self.k2],
                component=j,
                Robin_const=self.sqrtRobin_const**2,
                tau=self.tau,
                noFactor=True,
            )[0, 0, :]
        return y

    def InnerProd(
        self, x, A, y
    ):  ### TODO: accelerate in case of A matrix (block-diagonal)
        xAy = np.array([y[k].conj() @ A[k] @ x[k] for k in range(len(self.grid_k2))])
        # imag_err = np.abs(xAy.imag).max()
        # if imag_err > 1.e-6:
        #     raise Exception('imag error = {0}'.format(imag_err))
        ### NOTE: inner product with DA is not supposed to pass this assertion, since we have (f, B*df) + (df, B*f),which we replace with 2*Re(f, B*df)
        xAy = xAy.real
        xAy = self.Quad_k2(xAy)
        return xAy

    def Quad_k2(self, f):  ### integration over k2
        return 0.5 * (f[1:] + f[:-1]) @ self.dk2

    # ------------------------------------------------------------------------------------------
    # Post-treatment
    # ------------------------------------------------------------------------------------------

    def plot(self, dynamic=False, liftime=True):
        if dynamic:
            ion()
        else:
            ioff()

        if not hasattr(self, "fig"):
            # self.fig = figure('One-point spectra')
            # self.fig.clf()
            # ax = self.fig.gca()
            self.fig, ax = subplots(
                nrows=1, ncols=2, num="Calibration", clear=True, figsize=[20, 10]
            )
            self.ax = ax
            k1, z = zip(*self.DataPoints)

            ### Subplot 1: One-point spectra
            ax[0].set_title("One-point spectra")
            self.lines_SP_model = [None] * self.vdim
            self.lines_SP_data = [None] * self.vdim
            for i in range(self.vdim):
                (self.lines_SP_model[i],) = ax[0].plot(
                    k1, self.SP[:, i, i], "-", label="F{0:d} model".format(i)
                )
            for i in range(self.vdim):
                (self.lines_SP_data[i],) = ax[0].plot(
                    k1, self.DataValues[:, i, i], "--", label="F{0:d} data".format(i)
                )
            ax[0].legend()
            ax[0].set_aspect(3 / 4)
            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[0].set_xlabel(r"$k_1$")
            ax[0].set_ylabel(r"$F_i$")

            ### Subplot 1: One-point spectra
            ax[1].set_title("Eddy liftime")
            # k2 = self.grid_k2
            # k  = np.sqrt(k1**2 + k2**2)
            k = np.array(k1)
            self.tau_model = self.EddyLifetime.eval(k, 0 * k).flatten()
            # self.tau_ref   = k**(-2/3)
            self.tau_ref = MannEddyLifetime(k)
            (self.lines_LT_model,) = ax[1].plot(
                k, self.tau_model, "-", label=r"$\tau_{model}$"
            )
            (self.lines_LT_ref,) = ax[1].plot(
                k, self.tau_ref, "--", label=r"$\tau_{ref}$"
            )
            ax[1].legend()
            ax[1].set_aspect(3 / 4)
            # ax[1].set_xscale('log')
            # ax[1].set_yscale('log')
            ax[1].set_xlabel(r"$k$")
            ax[1].set_ylabel(r"$\tau$")

            self.fig.show()
            pause(0.1)

        k1, z = zip(*self.DataPoints)
        for i in range(self.vdim):
            self.lines_SP_model[i].set_xdata(k1)
            self.lines_SP_model[i].set_ydata(self.SP[:, i, i])
            # self.lines_SP_data[i].set_xdata(k1)
            # self.lines_SP_data[i].set_ydata(self.DataValues[:,i,i])
        k = np.array(k1)
        self.tau_model = self.EddyLifetime.eval(k, 0 * k).flatten()
        self.lines_LT_model.set_xdata(k1)
        self.lines_LT_model.set_ydata(self.tau_model)
        # self.ax[1].autoscale()
        # self.ax[1].set_aspect(3/4)
        # self.ax[1].update({})

        if dynamic:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            plt.show()


###################################################################################################
###################################################################################################

if __name__ == "__main__":
    """TEST by method of manufactured solutions"""

    pass
