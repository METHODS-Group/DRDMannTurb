import numpy as np
from collections.abc import Iterable
from .ode_solve import ode_solve
from ..RationalApproximation import compute_RationalApproximation_AAA
from ..RationalApproximation import compute_RationalApproximation_AAA_new
from time import time
import os
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

class fde_solve:
    '''
    solves (1+\mathcal{L})^{\alpha}\mathcal{L}^{\beta}\psi = f in [0,H]
                                                      \psi = 0 at z=0
                                                  d\psi/dz = 0 at z=H
    '''
    def __init__(self, dof, alpha, coef, domain_height=1, t=0, Robin_const=None, beta=None, z_grid=None):
        self.dof = dof
        self.alpha = alpha
        self.beta = beta
        self.coef = coef
        self.domain_height = domain_height
        self.t = t
        self.Robin_const=Robin_const
        self.z_grid = z_grid
        self.reset_ode(coef)
        self.separate = False
        # try:
        #     self.ode_solve = [ ode_solve(dof, coefj, domain_height=self.domain_height) for coefj in coef ]
        # except:
        #     self.ode_solve = ode_solve(dof, coef, domain_height=self.domain_height)
            
        # self.c, self.d = compute_RationalApproximation_AAA_new(alpha, beta, nPoints=1000)
        if not self.separate:
            self.c, self.d = compute_RationalApproximation_AAA_new(alpha, beta, nPoints=1000)
            # self.d = self.d - 1
            self.nModes = self.c.size
        else:
            self.c1, self.d1 = compute_RationalApproximation_AAA(5/12, nPoints=1000)
            self.nModes1 = self.c1.size
            self.c2, self.d2 = compute_RationalApproximation_AAA(1.0, nPoints=1000)
            self.nModes2 = self.c2.size
        self.grid = self.ode_solve.grid[:]
        self.h = self.ode_solve.grid.h
        if self.beta is None:
            self.kappa_alpha = self.coef(self.grid)**self.alpha
        else:
            self.kappa_alpha = self.coef(self.grid)**(self.alpha+self.beta)

    def reset_ode(self, coef):
        self.anisotrop = True if isinstance(coef, Iterable) else False
        self.ode_solve = ode_solve(self.dof, coef, domain_height=self.domain_height, grid=self.z_grid)
        self.apply_sqrtMass = self.ode_solve.apply_sqrtMass
        self.apply_Mass = self.ode_solve.apply_Mass
        # if isinstance(coef, Iterable):
        #     self.ode_solve = [ ode_solve(self.dof, coefj, domain_height=self.domain_height, grid=self.z_grid) for coefj in coef ]
        #     self.anisotrop = True
        #     self.apply_sqrtMass = self.ode_solve[0].apply_sqrtMass
        #     self.apply_Mass = self.ode_solve[0].apply_Mass
        # else:
        #     self.ode_solve = ode_solve(self.dof, coef, domain_height=self.domain_height, grid=self.z_grid)
        #     self.anisotrop = False
        #     self.apply_sqrtMass = self.ode_solve.apply_sqrtMass
        #     self.apply_Mass = self.ode_solve.apply_Mass
        

    def reset_kappa(self, coef):
        try:
            det = 1
            for coefj in coef:
                det = det * coefj(self.grid)
            det = det**(1/len(coef))
        except:
            det = coef(self.grid)
        if self.beta is None:
            self.kappa_alpha = det**self.alpha
        else:
            self.kappa_alpha = det**(self.alpha+self.beta)

    def reset_parameters(self, coef=None, t=None, Robin_const=None):
        self.Robin_const=Robin_const
        if coef is not None:
            self.coef = coef
            self.reset_ode(coef)
            self.reset_kappa(coef)
            # self.ode_solve = ode_solve(self.dof, coef, domain_height=self.domain_height)
            # if self.beta is None:
            #     self.kappa_alpha = self.coef(self.grid)**self.alpha
            # else:
            #     self.kappa_alpha = self.coef(self.grid)**(self.alpha+self.beta)


    def reset_jac(self, grad_coef):
        self.grad_term1 = [ self.alpha * der_coef(self.grid)/self.coef(self.grid) for der_coef in grad_coef ]
        self.gradA = [ ode_solve(self.dof, der_coef, domain_height=self.domain_height) for der_coef in grad_coef ]

        self.e1 = np.zeros(self.dof)
        self.e1[0] = 1

    def __call__(self, f, k1, k2, **kwargs):
        self.t = kwargs.get('t', 0)
        self.Robin_const = kwargs.get('Robin_const', None)
        # if self.Robin_const == 'infty':
        #     self.Robin_const = np.infty
        self.adjoint   = kwargs.get('adjoint', False)
        jac            = kwargs.get('jac', False)
        grad_coef      = kwargs.get('grad_coef', False)
        self.component = kwargs.get('component', 0)
        self.mode      = kwargs.get('mode', None)
        self.kwargs    = kwargs
        if 'Robin_const' in self.kwargs.keys():
            self.kwargs.__delitem__('Robin_const')        
        noL2factor = kwargs.get('noL2factor', False)

        if f is not None:
            if np.linalg.norm(f) == 0:
                return np.zeros([self.grid.size])

        # t0=time()
        self.rhs = 1*f  ### variant of copying vector (important fot not modifying the input)
        self.k = (k1, k2)
        if (not self.adjoint) and (not noL2factor):
            self.rhs *= self.kappa_alpha
        # func = lambda arg: self.ode_solve(arg[1], arg[0]*f, k1, k2, Robin_const=self.Robin_const)
        # modes = list(self.Pool.map(func, zip(self.c, self.d)))

        if self.mode is not None:
            psi_n_approx = func((self.mode, self))
        else:
            if not self.separate:
                args = list(enumerate((self,)*self.nModes))
                # self.modes = np.array(list(Pool().map(func, args)))
                self.modes = np.array(list(map(func, args)))
                psi_n_approx = self.modes.T @ self.c #np.sum(*, axis=0)
            else:
                # if self.anisotrop:
                #     psi_n_approx = self.ode_solve[self.component]( 0, psi_n_approx, k1, k2, Robin_const=self.Robin_const, **self.kwargs)
                # else:
                #     psi_n_approx = self.ode_solve( 0, psi_n_approx, k1, k2, Robin_const=0, **self.kwargs)

                self.c, self.d, self.nModes = self.c1, self.d1, self.nModes1
                args = list(enumerate((self,)*self.nModes))
                self.modes = np.array(list(map(func, args)))
                psi_n_approx = self.modes.T @ self.c

                if self.Robin_const is np.infty:
                    self.Robin_const = -1
                self.rhs = self.apply_Mass(psi_n_approx)

                self.c, self.d, self.nModes = self.c2, self.d2, self.nModes2
                args = list(enumerate((self,)*self.nModes))
                self.modes = np.array(list(map(func, args)))
                psi_n_approx = self.modes.T @ self.c

        if self.adjoint and (not noL2factor):
            psi_n_approx *= self.kappa_alpha
            # psi_n_approx = self.apply_sqrtMass(self.kappa_alpha * psi_n_approx) ### this is applied in Sampling_method now
        # print('Time FDE:', time()-t0)

        if jac and self.adjoint:
            ncomp = len(grad_coef)+1
            grad  = np.zeros(ncomp, dtype=psi_n_approx.dtype)
            self.Bf_conj = self.kappa_alpha*psi_n_approx.conj()
            self.modes_jac = list(map(func_jac, args))

            ### derivative wrt Robin const
            grad[-1] = np.sum(list(map(func_bc, args)), axis=0)

            ### other derivatives
            nOtherDer = grad.size-1
            if nOtherDer>0: ff = np.abs(psi_n_approx)**2
            for comp in range(nOtherDer):
                self.comp = comp
                grad[comp] = -np.sum(list(map(func_assemble, args)), axis=0)
                grad[comp] += np.sum(self.grad_term1[comp]*ff)


            ### legacy
            # for comp, der_coef in enumerate(grad_coef):
            #     self.gradA = ode_solve(self.dof, der_coef, t=self.t, domain_height=self.domain_height)
            #     self.modes_jac = list(map(func_jac, args))
            #     grad[...,comp] = np.sum(self.modes_jac, axis=0)
            #     grad[...,comp] *= -self.coef(grid)**self.alpha
            #     grad[...,comp] += self.alpha * der_coef(grid)/self.coef(grid) * psi_n_approx
            return psi_n_approx, grad
        else:
            return psi_n_approx


def func(args):
    i, self = args
    # if self.anisotrop:
    #     return self.ode_solve[self.component](  self.d[i], self.rhs, self.k[0], self.k[1], Robin_const=self.Robin_const, **self.kwargs)
    # else:
    return self.ode_solve(  self.d[i], self.rhs, self.k[0], self.k[1], Robin_const=self.Robin_const, **self.kwargs)

def func_jac(args):
    i, self = args
    return self.ode_solve(  self.d[i], self.Bf_conj, self.k[0], self.k[1],
                            t=self.t, Robin_const=self.Robin_const, adjoint=False)  

def func_bc(args):
    i, self = args
    return -self.modes_jac[i][0]*self.modes[i][0]/self.h

def func_assemble(args):
    i, self = args
    gradA_mode = self.gradA[self.comp].apply_matvec(-1, self.modes[i], self.k[0], self.k[1])
    return np.sum(self.modes_jac[i]*gradA_mode)


### legacy

# def func_jac1(args):
#     i, self = args
#     mode = self.gradA.apply_matvec(-1, self.modes[i], self.k[0], self.k[1])
#     return self.ode_solve(  self.d[i], mode, self.k[0], self.k[1],
#                             Robin_const=self.Robin_const, adjoint=self.adjoint)

# def func_bc1(args):
#     i, self = args
#     return self.modes[i][0]*self.ode_solve(  self.d[i], self.e1, self.k[0], self.k[1],
#                                             Robin_const=self.Robin_const, adjoint=self.adjoint)



# def func_bc_aux(args):
#     i, self = args
#     return self.ode_solve(  self.d[i], self.e1, self.k[0], self.k[1],
#                             Robin_const=self.Robin_const, adjoint=self.adjoint)


# def func(i, self, out):
#     # i, self, out = args
#     out[i] = self.ode_solve(self.d[i], self.c[i]*self.rhs, self.k[0], self.k[1], Robin_const=self.Robin_const)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    '''TEST by method of manufactured solutions'''

    ## TODO