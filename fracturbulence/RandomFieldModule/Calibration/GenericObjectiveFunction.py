from math import *
import numpy as np
import scipy.optimize
from collections.abc import Iterable
import torch
from pylab import *
from time import time, sleep
from multiprocessing import Process
from matplotlib.animation import FuncAnimation
import matplotlib.ticker


from .EddyLifetime import EddyLifetime


###################################################################################################
#  Generic Objective function
###################################################################################################


class GenericObjectiveFunction:
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get("verbose", False)
        self.tol = kwargs.get("tol", 1.0e-3)

        self.dev = kwargs.get("dev", False)  ### development mode (testing)
        if self.dev:
            self.verbose = True

        self.ModelObj = args[0]
        self.DataPoints, self.DataValues = args[1]

        ### Eddy Life Time
        self.EddyLifetime = EddyLifetime(**kwargs)

        ### Dimensions and grids
        self.vdim = self.ModelObj.vdim
        self.nDataPoints = len(self.DataPoints)
        self.grid_k2 = np.r_[-np.logspace(-2, 3, 100)[::-1], 0, np.logspace(-2, 3, 100)]
        self.dk2 = np.diff(self.grid_k2)  # [1:]-self.grid_k2[:-1]

        DP = np.array(self.DataPoints)
        self.grid_k1 = DP[:, 0] if DP.ndim > 1 else DP

    ###------------------------------------------------
    ### Parameters
    ###------------------------------------------------

    @property
    def NN_parameters(self):
        return self.EddyLifetime.parameters

    @NN_parameters.setter
    def NN_parameters(self, params):
        self.EddyLifetime.update_parameters(params)

    @property
    def Common_parameters(self):
        return np.array([])
        # NOTE: your code in the child class

    @Common_parameters.setter
    def Common_parameters(self, params):
        # NOTE: your code in the child class
        pass

    @property
    def All_parameters(self):
        return self.Common_parameters.tolist() + self.NN_parameters.tolist()

    @All_parameters.setter
    def All_parameters(self, params):
        assert len(params) == len(self.All_parameters)
        nNNparams = len(self.NN_parameters)
        self.Common_parameters = params[:-nNNparams]
        self.NN_parameters = params[-nNNparams:]

    def get_parameters(self):
        return self.All_parameters

    def set_parameters(self, theta):
        self.All_parameters = theta

    ###------------------------------------------------
    ### Computations
    ###------------------------------------------------

    def __call__(self, theta):
        self.update(theta)
        if self.verbose:
            print("Function call: J = {0}".format(self.J))
            print("theta =\n", theta)
        self.plot(dynamic=True)
        return self.J

    def Jacobian(self, theta):
        self.update(theta, jac=True)
        if self.verbose:
            print("Jacobian call: DJ =\n", self.DJ)
        return self.DJ

    def initialize(self):
        # NOTE: your code in the child class
        pass

    def update(self, theta, jac=False):
        self.SP = np.zeros_like(self.DataValues)
        self.J = 0
        if jac:
            self.DJ = 0
        # NOTE: your code in the child class

    def finalize(self):
        # NOTE: your code in the child class
        pass

    def default_parameters(self):
        return 0

    ###------------------------------------------------
    ### Tools
    ###------------------------------------------------

    def Quad_k2(self, f):  ### integration over k2
        return 0.5 * (f[1:] + f[:-1]) @ self.dk2

    ###------------------------------------------------
    ### Post-treatment and Export
    ###------------------------------------------------

    def plot(self, dynamic=False):
        if dynamic:
            ion()
        else:
            ioff()

        if not hasattr(self, "fig"):
            self.fig, self.ax = subplots(
                nrows=1, ncols=2, num="Calibration", clear=True, figsize=[20, 10]
            )
            # k1, z = zip(*self.DataPoints)
            k1 = self.grid_k1

            ### Subplot 1: One-point spectra
            self.ax[0].set_title("One-point spectra")
            self.lines_SP_model = [None] * self.vdim
            self.lines_SP_data = [None] * self.vdim
            for i in range(self.vdim):
                (self.lines_SP_model[i],) = self.ax[0].plot(
                    k1, self.SP[:, i, i], "o-", label="F{0:d} model".format(i)
                )
            for i in range(self.vdim):
                (self.lines_SP_data[i],) = self.ax[0].plot(
                    k1, self.DataValues[:, i, i], "--", label="F{0:d} data".format(i)
                )
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

            ### Subplot 1: One-point spectra
            self.ax[1].set_title("Eddy liftime")
            # k2 = self.grid_k2
            # k  = np.sqrt(k1**2 + k2**2)
            k = np.array(k1)
            # self.tau_model = self.EddyLifetime.eval(k, 0*k).flatten()
            # self.tau_model = self.tau
            self.tau_ref = k ** (-2 / 3)
            (self.lines_LT_model,) = self.ax[1].plot(
                k, self.tau_model, "-", label=r"$\tau_{model}$"
            )
            (self.lines_LT_ref,) = self.ax[1].plot(
                k, self.tau_ref, "--", label=r"$\tau_{ref}=|k|^{-\frac{2}{3}}$"
            )
            self.ax[1].legend()
            self.ax[1].set_aspect(3 / 4)
            self.ax[1].set_xscale("log")
            self.ax[1].set_yscale("log")
            self.ax[1].set_xlabel(r"$k$")
            self.ax[1].set_ylabel(r"$\tau$")
            self.ax[1].grid(which="both")

            # self.fig.show()
            # pause(0.1)

        # k1, z = zip(*self.DataPoints)
        for i in range(self.vdim):
            # self.lines_SP_model[i].set_xdata(k1)
            self.lines_SP_model[i].set_ydata(self.SP[:, i, i])
            # self.lines_SP_data[i].set_xdata(k1)
            # self.lines_SP_data[i].set_ydata(self.DataValues[:,i,i])
        # k = np.array(k1)
        # self.tau_model = self.EddyLifetime.eval(k, 0*k).flatten()
        # self.lines_LT_model.set_xdata(k1)
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
