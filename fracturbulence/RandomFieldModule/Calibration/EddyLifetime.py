import sys

sys.path.append(
    "/Users/bk/Work/Papers/Collaborations/2020_inletgeneration/code/source/"
)
sys.path.append("/home/khristen/Projects/Brendan/2019_inletgeneration/code/source")

from collections.abc import Callable, Iterable
from math import *

import matplotlib.pyplot as plt
import numpy as np
# from fracturbulence.RandomFieldModule.utilities.ode_solve import FEM_coefficient_matrix_generator, Grid1D
import scipy.fftpack as fft
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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
            out = 3.9 * self.scaling(out)  # out = const * k ** NN(k1,k2)
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
    def __init__(self, tau0=0.0, hidden_layer_size=16, noise_magnitude=1e-3, **kwargs):
        self.tau0 = tau0
        self.NN = NeuralNet(2, hidden_layer_size)
        NN_parameters = parameters_to_vector(self.NN.parameters())
        with torch.no_grad():
            self.parameters = NN_parameters.numpy()
        self.initialize_parameters_with_noise(noise_magnitude=noise_magnitude)

    def initialize_parameters_with_noise(self, noise_magnitude=1e-3):
        noise = noise_magnitude * np.random.randn(*self.parameters.shape)
        self.update_parameters(noise)

    def update_parameters(self, param_vec):
        assert len(param_vec) > 1
        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(param_vec, dtype=torch.float64)
        vector_to_parameters(param_vec, self.NN.parameters())
        NN_parameters = parameters_to_vector(self.NN.parameters())
        with torch.no_grad():
            self.parameters = NN_parameters.numpy()

    def eval(self, k1, k2=0):
        arg = self.format_input(k1, k2)
        k = np.sqrt(k1**2 + k2**2)
        with torch.no_grad():
            if callable(self.tau0):
                return self.tau0(k) + self.NN(arg).numpy().flatten()
            else:
                return self.tau0 + self.NN(arg).numpy().flatten()

    def eval_deriv(self, k1, k2):
        self.NN.zero_grad()
        arg = self.format_input(k1, k2)
        self.NN(arg).backward()
        tmp = []
        for param in self.NN.parameters():
            tmp.append(param.grad.view(-1))
        dtau = torch.cat(tmp)
        return dtau.numpy()

    def format_input(self, *args):
        if np.isscalar(args[0]):
            Input = torch.tensor([args[0], args[1]], dtype=torch.float64).unsqueeze(0)
        else:
            Input = torch.tensor([args[0], args[1]], dtype=torch.float64).t()
        return Input

    # ''' input is the qp array and the scalar frequency '''
    # def eval_qp(self, qp, k1):
    #     k1 = float(k1)
    #     tau_at_k1 = np.vectorize(lambda z : self.eval(z,k1))
    #     with torch.no_grad():
    #         return tau_at_k1(qp)

    # ''' input is the qp array and the scalar frequency '''
    # def eval_deriv_qp(self, qp, k1):
    #     dtau_at_k1 = np.vectorize(lambda z : self.eval_deriv(z,k1),otypes=[np.ndarray])
    #     tmp = dtau_at_k1(qp)
    #     # convert 2D array of arrays to 3D array
    #     tmp = np.array([e.tolist() for e in tmp.flatten()]).reshape(tmp.shape[0],tmp.shape[1],-1)
    #     # convert 3D array to generator indexing 2D arrays
    #     return (tmp[:,:,k] for k in range(tmp.shape[2]))

    def Initialize(self, **kwargs):
        print("\nInitializing NeuralNet...")

        if "func" in kwargs.keys():
            func = kwargs.get("func")
            x = kwargs.get("x", None)
            method = kwargs.get("optimizer", "LBFGS")
            lr = kwargs.get("learning_rate", 1e-2)
            tol = kwargs.get("tol", 1e-3)
            show = kwargs.get("show", False)

            if x is None:
                x = np.logspace(-1, 2, 1000)
            x = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)
            xx = torch.cat((x, 0 * x), dim=1)
            y = func(x)

            loss_fn = torch.nn.MSELoss(reduction="sum")
            # loss_fn = torch.nn.NLLLoss(reduction='sum')

            ### Optimizer
            if method == "LBFGS":
                method = torch.optim.LBFGS
            elif method == "RMSprop":
                method = torch.optim.RMSprop

            def closure():
                optimizer.zero_grad()
                yy = self.NN(xx)
                loss = loss_fn(yy, y)
                loss.backward()
                return loss

            def plot():
                yy = self.NN(xx)
                loss = loss_fn(yy, y)
                print("   ", loss.item())
                plt.plot(x.detach().numpy(), y.detach().numpy(), label="target")
                plt.plot(x.detach().numpy(), yy.detach().numpy(), label="nn")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.show()

            plot()
            old_params = parameters_to_vector(self.NN.parameters())
            for lr_j in lr * 0.1 ** np.arange(10):
                optimizer = method(self.NN.parameters(), lr=lr)
                for t in range(100):
                    optimizer.step(closure)
                    if t % 5 == 0:
                        plot()
                current_params = parameters_to_vector(self.NN.parameters())
                if any(np.isnan(current_params.data.cpu().numpy())):
                    print("Optimization diverged. Rolling back update...")
                    vector_to_parameters(old_params, self.NN.parameters())
                else:
                    break
            plot()

            with torch.no_grad():
                self.parameters = parameters_to_vector(self.NN.parameters()).numpy()

        else:  ### initialize with noise
            noise_magnitude = kwargs.get("noise_magnitude", 1.0e-3)
            self.initialize_parameters_with_noise(noise_magnitude=noise_magnitude)

        print(
            (
                "Initial NN parameters = ["
                + ", ".join(["{}"] * len(self.parameters))
                + "]\n"
            ).format(*self.parameters)
        )
        return self.parameters


"""

    Eddy lifetime grid class

    Comments:
    Used to store value and derivative information on a grid of frequencies in x and y

"""


class EddyLifetimeGrid(EddyLifetime):
    def __init__(self, frequencies, hidden_layer_size=5, tau0=0, noise_magnitude=1e-3):
        assert frequencies.shape[0] == 2

        self.frequencies = frequencies
        super().__init__(
            tau0=tau0,
            hidden_layer_size=hidden_layer_size,
            noise_magnitude=noise_magnitude,
        )

    def update_parameters(self, param_vec):
        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(param_vec, dtype=torch.float64)
        vector_to_parameters(param_vec, self.NN.parameters())
        NN_parameters = parameters_to_vector(self.NN.parameters())
        with torch.no_grad():
            self.parameters = NN_parameters.numpy()
        self.values = self.fill_tau()
        self.derivatives = self.fill_tau_derivatives()

    def fill_tau(self):
        M = self.frequencies.shape[1]
        values = np.empty((M, M))
        for i, k1 in enumerate(self.frequencies[0, :]):
            for j, k2 in enumerate(self.frequencies[1, :]):
                values[i, j] = self.eval(k1, k2)
        return values

    def fill_tau_derivatives(self):
        M = self.frequencies.shape[1]
        derivatives = np.empty((M, M, len(self.parameters)))
        for i, k1 in enumerate(self.frequencies[0, :]):
            for j, k2 in enumerate(self.frequencies[1, :]):
                derivatives[i, j, :] = self.eval_deriv(k1, k2)[:]
        return derivatives


# """

#     General class for computing a matrix and its derivatives for the specific frequency (k1,k2)

#     Comments:
#     Made inputs similar to ode_solve

# """
# class DerivativeMatrixGenerator:

#     def __init__(self, dof, EddyLifetime, domain_height=1):
#         self.grid = Grid1D(dof, upper_bound=domain_height)
#         self.tau = EddyLifetime

#     def update_parameters(self,param_vec):
#         self.tau.update_parameters(param_vec)

#     def assemble_matrix(self, **kwargs):
#         return 0

#     def assemble_derivative(self, k1, k2):
#         return 0

# """

#     Assembles A and dA/dparams for the specific frequency (k1,k2)

# """
# class DerivativeStiffnessMatrixGenerator(DerivativeMatrixGenerator):

#     def quadrature_points(self):
#         qp = np.zeros([2, len(self.grid)-1])
#         qp[0,:] = (self.grid[1:] - self.grid.h/2) - 1/sqrt(3) * self.grid.h/2
#         qp[1,:] = (self.grid[1:] - self.grid.h/2) + 1/sqrt(3) * self.grid.h/2
#         return qp

#     def __init__(self, dof, coef, EddyLifetime, domain_height=1):
#         super().__init__(dof, EddyLifetime, domain_height)
#         if isinstance(coef, Iterable):
#             if len(coef)==1:
#                 self.coef = [coef[0]] * 3
#             elif len(coef)==2:
#                 self.coef = [ coef[0], coef[0], coef[1] ]
#             else:
#                 self.coef = coef
#         else:
#             self.coef = [coef] * 3
#         self.qp = self.quadrature_points()
#         self.Mass  = self.mass_matrix(self.func_const1())
#         self.Mass1 = self.mass_matrix(self.coef[0])
#         self.Mass2 = self.mass_matrix(self.coef[1])
#         self.Mass3 = self.mass_matrix(self.coef[2])
#         self.Cross = self.crossterm_diffusion_matrix(self.coef[2])
#         self.Diff  = self.diffusion_matrix(self.coef[2])

#     @staticmethod
#     def func_const1():
#         return (lambda z : 1)

#     def assemble_matrix(self, d, k1, k2):
#         with torch.no_grad():
#             tau = self.tau.eval(k1,k2)
#         M = self.Mass
#         M1 = self.Mass1
#         M2 = self.Mass2
#         M3 = self.Mass3
#         C = self.Cross
#         D = self.Diff
#         return (d+1)*M + (k1**2)*M1 + (k2**2)*M2 + ((tau*k1)**2)*M3 + (tau*k1)*C + D

#     def assemble_derivative(self, k1, k2):
#         with torch.no_grad():
#             tau = self.tau.eval(k1,k2)
#         dtau = self.tau.eval_deriv(k1,k2)
#         M3 = self.Mass3
#         C = self.Cross
#         dAdtau = 2*tau*(k1**2)*M3 + k1*C
#         return [dAdtau * dtau_i for dtau_i in dtau]

#     def mass_matrix(self, reaction_function):
#         '''creates the mass matrix in matrix diagonal ordered form'''
#         M = np.zeros((3,len(self.grid)))
#         fun_qp = reaction_function(self.qp)
#         if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
#         w = 1/2
#         x1 = (1-1/np.sqrt(3))/2
#         x2 = (1+1/np.sqrt(3))/2
#         coeff_element_0 = (fun_qp[0,:]*(x1**2) + fun_qp[1,:]*(x2**2)) * self.grid.h * w
#         coeff_element_1 = (fun_qp[0,:]*(x1*(1-x1)) + fun_qp[1,:]*(x2*(1-x2))) * self.grid.h * w
#         M[0,1:]  = coeff_element_1
#         M[1,:-1] = coeff_element_0
#         M[1,1:] += coeff_element_0
#         M[2,:-1] = M[0,1:]
#         return M

#     def crossterm_diffusion_matrix(self, crossterm_diffusion_function):
#         '''creates the cross-term contributions to the diffusion matrix in matrix diagonal ordered form
#            NOTE that this contribution is ONLY active when t > 0 '''
#         D = np.zeros((3,len(self.grid)))
#         fun_qp = crossterm_diffusion_function(self.qp)
#         if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
#         w = 1/2
#         x1 = (1-1/np.sqrt(3))/2
#         x2 = (1+1/np.sqrt(3))/2
#         coeff_element = 2*(fun_qp[0,:]*x1 + fun_qp[1,:]*x2) * w
#         D[0,1:]  = -coeff_element
#         D[2,:-1] = -D[0,1:]
#         return 1j*D

#     def diffusion_matrix(self, diffusion_function):
#         '''creates the diffusion matrix in matrix diagonal ordered form'''
#         D = np.zeros((3,len(self.grid)))
#         fun_qp = diffusion_function(self.qp)
#         if np.isscalar(fun_qp): fun_qp *= np.ones_like(self.qp)
#         w = 1/2
#         coeff_element = fun_qp.sum(axis=0)*w/self.grid.h
#         D[0,1:]  = -coeff_element
#         D[1,:-1] = coeff_element
#         D[1,1:] += coeff_element
#         D[2,:-1] = D[0,1:]
#         return D

# """

#     Assembles B and dB/dparams for the specific frequency (k1,k2)

# """
# class DerivativeCovarianceMatrixGenerator(DerivativeMatrixGenerator):

#     def __init__(self, dof, EddyLifetime, domain_height=1):
#         super().__init__(dof, EddyLifetime, domain_height)
#         self.compute_k3()
#         self.compute_fourier_basis()

#     def compute_k3(self):
#         N = len(self.grid)
#         L = self.grid.upper_bound - self.grid.lower_bound
#         self.k3 = (2*pi/L)*(N*fft.fftfreq(N))

#     def compute_fourier_basis(self):
#         assert(isinstance(self.grid.h, float))
#         h = self.grid.h
#         M = len(self.grid)
#         phi_hat = np.zeros((M,M),dtype=complex)
#         for l in range(M):
#             phi_hat[l,:] = np.sinc(h*self.k3/(2*np.pi))**2 * np.exp(-1j * h * l * self.k3) * h / np.sqrt(2*np.pi)
#         self.phi_hat = phi_hat

#     def assemble_matrix(self, k1, k2):

#         assert(isinstance(k1, float))
#         assert(isinstance(k2, float))

#         with torch.no_grad():
#             tau = self.tau.eval(k1,k2)
#         k3 = self.k3
#         k30  = k3 + tau*k1

#         kk = k1**2 + k2**2 + k3**2
#         kk0 = k1**2 + k2**2 + k30**2

#         s = k1**2 + k2**2
#         C1  =  tau * k1**2 * (kk0 - 2 * k30**2 + tau * k1 * k30) / (kk * s)
#         tmp =  tau * k1 * np.sqrt(s) / (kk0 - k30 * k1 * tau)
#         C2  =  k2 * kk0 / s**(3/2) * np.arctan (tmp)

#         zeta1_by_zeta3 =  (C1 - k2/k1 * C2)*kk/kk0
#         zeta2_by_zeta3 =  (k2/k1 *C1 + C2)*kk/kk0
#         one_by_zeta3   =  kk/kk0

#         zeta1_by_zeta3[np.isnan(zeta1_by_zeta3)] = 0
#         zeta2_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0
#         one_by_zeta3[np.isnan(zeta2_by_zeta3)] = 0

#         M = len(self.grid[:])
#         G = np.zeros((3,3,M))
#         G[0,0,:] = 1;        G[0,1,:] = 0;        G[0,2,:] = -zeta1_by_zeta3
#         G[1,0,:] = 0;        G[1,1,:] = 1;        G[1,2,:] = -zeta2_by_zeta3
#         G[2,0,:] = G[0,2,:]; G[2,1,:] = G[1,2,:]; G[2,2,:] = zeta1_by_zeta3**2 + zeta2_by_zeta3**2 + one_by_zeta3**2

#         # NOTE: Can be optimized (B11 and B22 are mass matrices, B12 is zero)
#         B = np.zeros((3*M,3*M),dtype=complex)
#         phi_hat = self.phi_hat
#         for i in range(3):
#             for j in range(3):
#                 for l in range(M):
#                     for m in range(M):
#                         B[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * G[i,j,:] * phi_hat[m,:])

#         return B

#     '''create list of FEM matrices in so-called "matrix diagonal ordered form", indexed by the NN parameters '''
#     def assemble_derivative(self, k1, k2):

#         assert(isinstance(k1, float))
#         assert(isinstance(k2, float))

#         with torch.no_grad():
#             tau = self.tau.eval(k1,k2)
#         k3 = self.k3
#         k30  = k3 + tau*k1

#         kk = k1**2 + k2**2 + k3**2
#         kk0 = k1**2 + k2**2 + k30**2
#         s = k1**2 + k2**2

#         # C1 and derivative
#         C1  =  tau * k1**2 * (kk0 - 2 * k30**2 + tau * k1 * k30) / (kk * s)
#         dC1dtau = k1**2 * (kk0 - 2 * k30**2 + 2 * tau * k1 * k30) / (kk * s)

#         a = k1 * np.sqrt(s)
#         b = kk0
#         c = k30 * k1
#         tmp1 = np.arctan( a * tau  / (b - c * tau) )
#         tmp2 = a * b / ( (a * tau)**2  + (c * tau - b)**2)
#         # C2 and derivative
#         C2  =  k2 * kk0 / s**(3/2) * tmp1
#         dC2dtau = k2 * kk0 / s**(3/2) * tmp2

#         # zetas and derivatives
#         zeta1 = (C1 - k2/k1 * C2)
#         zeta2 = (k2/k1 * C1 + C2)
#         dzeta1dtau = (dC1dtau - k2/k1 * dC2dtau)
#         dzeta2dtau = (k2/k1 * dC1dtau + dC2dtau)

#         # non-zero components of dGdtau
#         dG13dtau = - dzeta1dtau * kk/kk0
#         dG23dtau = - dzeta2dtau * kk/kk0
#         dG33dtau = 2 * ( zeta1 * dzeta1dtau + zeta2 * dzeta2dtau ) * (kk/kk0)**2

#         dG13dtau[np.isnan(dG13dtau)] = 0
#         dG23dtau[np.isnan(dG23dtau)] = 0
#         dG33dtau[np.isnan(dG33dtau)] = 0

#         # dBdtau
#         M = len(self.grid[:])
#         dBdtau = np.zeros((3*M,3*M),dtype=complex)
#         phi_hat = self.phi_hat

#         i=2; j=0
#         for l in range(M):
#             for m in range(l,M):
#                 dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG13dtau * phi_hat[m,:])

#         i=2; j=1
#         for l in range(M):
#             for m in range(l,M):
#                 dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG23dtau * phi_hat[m,:])

#         i=2; j=2
#         for l in range(M):
#             for m in range(l,M):
#                 dBdtau[i*M + l,j*M + m] = np.sum(phi_hat[l,:] * dG33dtau * phi_hat[m,:])

#         dBdtau = dBdtau + dBdtau.T - np.diag(dBdtau.diagonal())

#         dtau = self.tau.eval_deriv(k1,k2)
#         return [dBdtau * dtau_i for dtau_i in dtau]


############################################################################
############################################################################

if __name__ == "__main__":
    tau0 = 0.0
    hidden_layer_size = 1
    # noise_magnitude = 1.0
    frequencies = np.array([[1.0, -1.0], [-2.0, 4.0]])
    # EddyLifetimeGrid = EddyLifetimeGrid(frequencies, tau0=tau0, hidden_layer_size=hidden_layer_size, noise_magnitude=noise_magnitude)
    # print(EddyLifetimeGrid.parameters)
    # print(EddyLifetimeGrid.values)
    # print(EddyLifetimeGrid.derivatives)

    # params = EddyLifetimeGrid.parameters
    # noise = np.random.randn(len(params))
    # EddyLifetimeGrid.update_parameters(noise)
    # print(EddyLifetimeGrid.parameters)
    # print(EddyLifetimeGrid.values)
    # print(EddyLifetimeGrid.derivatives)

    h = 1e-3
    ind = 0
    EddyLifetime = EddyLifetime(tau0=1, hidden_layer_size=hidden_layer_size)
    params = EddyLifetime.parameters
    params = np.random.randn(len(params))
    EddyLifetime.update_parameters(params)
    dp = np.zeros_like(params)
    dp[ind] = h

    k1 = 1.0
    k2 = -1.0
    tau0 = EddyLifetime.eval(k1, k2)
    dtau = EddyLifetime.eval_deriv(k1, k2)  # [ind]
    EddyLifetime.update_parameters(params + dp)
    tau1 = EddyLifetime.eval(k1, k2)
    FDtau = (tau1 - tau0) / h
    print("params = ", EddyLifetime.parameters)
    print("tau0 = ", tau0)
    print("dtau = ", dtau)
    print("FDtau = ", FDtau)
    print("error = ", np.abs(dtau[ind] - FDtau))

    # L1 = lambda z : z
    # L2 = lambda z : z**2
    # L3 = lambda z : z**3
    # coef = kappa = [ np.vectorize(lambda z: L1(z)**2), np.vectorize(lambda z: L2(z)**2), np.vectorize(lambda z: L3(z)**2) ]

    # domain_height = 0.9
    # dof = 2**6
    # DerivativeStiffnessMatrixGenerator = DerivativeStiffnessMatrixGenerator(dof, coef, EddyLifetime, domain_height=domain_height)
    # DerivativeCovarianceMatrixGenerator = DerivativeCovarianceMatrixGenerator(dof, EddyLifetime, domain_height=domain_height)

    # d = 0.0
    # k1 = 1.0
    # k2 = np.pi
    # A = DerivativeStiffnessMatrixGenerator.assemble_matrix(d, k1, k2)
    # DA = DerivativeStiffnessMatrixGenerator.assemble_derivative(k1, k2)
    # B = DerivativeCovarianceMatrixGenerator.assemble_matrix(k1, k2)
    # DB = DerivativeCovarianceMatrixGenerator.assemble_derivative(k1, k2)

    # print(len(DerivativeStiffnessMatrixGenerator.tau.parameters))
    # print(np.array(DA).shape)
    # print(np.array(DB).shape)
