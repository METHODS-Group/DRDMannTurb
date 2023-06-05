
from math import *
from pylab import *
import numpy as np
import scipy.optimize
import scipy.fftpack as fft
from scipy.special import gamma
import copy
from time import time
import matplotlib.pyplot as plt 

from ..utilities.common import Mv




###################################################################################################
#   Parameters expansion
###################################################################################################


class BasicExpansion:

    def __init__(self, **kwargs):        
        self.L_inf   = kwargs.get('L_inf',1)
        self.L_inf_2 = self.L_inf**2

    def update_L_inf(self, L_inf):     
        self.L_inf   = L_inf
        self.L_inf_2 = self.L_inf**2


    def __call__(self, z):
        return 0

    def plot(self):
        z = np.linspace(0,1,1000)
        y = sqrt(self(z)/self.L_inf_2)
        plt.plot(y,z)
        plt.xlabel('L(z)')
        plt.ylabel('z')
        return z, y

###################################################################################################

class ExponentialExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nTerms = kwargs.get('nTerms',0)
        self.c = kwargs.get('c', np.zeros((self.nTerms,)) )
        self.d = kwargs.get('d', np.zeros((self.nTerms,)) )
        self.nPars = self.c.size+self.d.size

        self.eval_vec = np.vectorize(self.eval)
        self.Gradient = [ ExpansionDerivative(self, comp) for comp in range(self.c.size+self.d.size) ]

    def __call__(self, z):
        return self.eval_vec(z)

    def update(self, params):
        params_iter = iter(params)
        self.c[:] = [ next(params_iter) for i in range(self.c.size) ]
        self.d[:] = [ next(params_iter) for i in range(self.d.size) ]

    def eval(self, z):
        if self.c.size == 0:
            return self.L_inf_2
        else:
            return self.L_inf_2 * (1 + np.sum(self.c*np.exp(-self.d**2*z/self.L_inf), axis=0))**2
        # return self.L_inf_2 * (1 + np.sum(self.c*np.exp(-self.d*z[:,None]), axis=1))

    def Derivative(self, z, comp):
        if comp < self.c.size:
            i = comp
            return self.L_inf_2*np.exp(-self.c[i]**2*(z+self.d[i]**2)) * 2*self.c[i]*(z+self.d[i]**2)
        elif comp < self.c.size+self.d.size:
            i = comp-self.c.size
            return self.L_inf_2 * np.exp(-self.c[i]**2*(z+self.d[i]**2)) * self.c[i]**2 * 2*self.d[i]
        else:
            raise Exception('Unkown derivative component!')



###################################################################################################

class PolyExpExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = kwargs.get('sigma', 0 )
        self.rho   = kwargs.get('rho', self.L_inf )
        nTerms= kwargs.get('nTerms',  2)
        self.poly  = kwargs.get('poly',  np.zeros(nTerms) )
        self.nPars = 2 + self.poly.size

        self.Gradient = [ ExpansionDerivative(self, comp) for comp in range(self.nPars) ]

    def update(self, params):
        params_iter = iter(params)
        self.sigma  = next(params_iter) or self.sigma
        self.rho    = next(params_iter) or self.rho
        self.poly[:]= [ next(params_iter) for i in range(self.poly.size)]

    def polyval(self, z):
        p = np.append(self.poly**2, 0)
        return np.polyval(p, z)

    def polyval_der1(self, z):
        p = np.append(self.poly**2, 0)
        p *= np.arange(p.size)[::-1]
        return np.polyval(p, z)


    def __call__(self, z):
        return self.L_inf_2 * (1 - self.sigma**2 * np.exp(-self.polyval(z/self.rho**2)))

    def Derivative(self, z, comp):
        if comp is 0:
            return -2*self.sigma * np.exp(-self.polyval(z/self.rho**2)) * self.L_inf_2
        elif comp is 1:
            return -self.sigma**2 * np.exp(-self.polyval(z/self.rho**2)) * self.L_inf_2 * (self.polyval_der1(z/self.rho**2) * 2/self.rho)
        elif comp >= 2:
            i = comp-2
            k = self.poly.size-i
            return -self.sigma**2 * np.exp(-self.polyval(z/self.rho**2)) * self.L_inf_2 * (-2*self.poly[i]*(z/self.rho**2)**k)
        else:
            raise Exception('Unkown derivative component!')



###################################################################################################

class PowerExpExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma = kwargs.get('sigma', 0 )
        self.rho   = kwargs.get('rho', self.L_inf )
        self.nu    = kwargs.get('nu',  0.5 )
        self.nPars = 3

        self.Gradient = [ ExpansionDerivative(self, comp) for comp in range(self.nPars) ]

    def update(self, params):
        params_iter = iter(params)
        self.sigma  = next(params_iter) or self.sigma
        self.rho    = next(params_iter) or self.rho
        self.nu     = next(params_iter) or self.nu

    def get_parameters(self):
        return 

    def __call__(self, z):
        return self.L_inf_2 * (1 - self.sigma**2 * np.exp(-(z/self.rho)**self.nu))

    def Derivative(self, z, comp):
        if comp is 0:
            return -2*self.sigma * np.exp(-(z/self.rho)**self.nu) * self.L_inf_2
        elif comp is 1:
            return -self.sigma**2 * np.exp(-(z/self.rho)**self.nu) * self.L_inf_2 * (self.nu*(z/self.rho)**(self.nu-1) / self.rho**2)
        elif comp is 2:
            return -self.sigma**2 * np.exp(-(z/self.rho)**self.nu) * self.L_inf_2 * ((z/self.rho)**(self.nu) * np.log(z/self.rho))
        else:
            raise Exception('Unkown derivative component!')


###################################################################################################

class MultiPoleExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nTerms = kwargs.get('nTerms',0)
        self.c = kwargs.get('c', np.zeros((self.nTerms,)) )
        self.d = kwargs.get('d', np.ones((self.nTerms,)) )
        self.nPars = self.c.size+self.d.size

        self.Gradient = [ ExpansionDerivative(self, comp) for comp in range(self.c.size+self.d.size) ]

    def update(self, params):
        params_iter = iter(params)
        self.c[:] = [ next(params_iter) for i in range(self.c.size) ]
        self.d[:] = [ next(params_iter) for i in range(self.d.size) ]

    def __call__(self, z):
        return self.L_inf_2 * ( 1 + np.sum(self.c/(z[:,None]/self.L_inf+self.d**2), axis=1) )**2
        # return self.L_inf_2 * ( 1 + np.sum(self.c/(z[:,None]+self.d**2), axis=1) )**2

    def Derivative(self, z, comp):
        if comp < self.c.size:
            i = comp
            return 1/self.L_inf/(z+self.d[i]**2/self.L_inf)
        elif comp < self.c.size+self.d.size:
            i = comp-self.c.size
            return -2*self.d[i]*self.c[i]/self.L_inf**2/(z+self.d[i]**2/self.L_inf)**2
        else:
            raise Exception('Unkown derivative component!')


###################################################################################################

class ExpansionDerivative:
    def __init__(self, ExpansionObj, comp):
        self.Expansion = ExpansionObj
        self.comp = comp

    def __call__(self, z):
        return self.Expansion.Derivative(z, comp=self.comp)


###################################################################################################

###################################################################################################
# 2nd paper expansions
###################################################################################################

class LogarithmicExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nPars = 1
        self.eval_vec = np.vectorize(self.eval)

    def __call__(self, z):
        return self.eval_vec(z)

    def update(self, params):
        params_iter = iter(params)
        self.a = next(params_iter)

    def eval(self, z):
        L = self.L_inf*np.log(self.a*z/self.L_inf + 1)
        return L**2

###################################################################################################

class FracPowerExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nPars = 2
        self.eval_vec = np.vectorize(self.eval)

    def __call__(self, z):
        return self.eval_vec(z)

    def update(self, params):
        params_iter = iter(params)
        self.a  = next(params_iter)**2
        self.nu = next(params_iter)**2

    def eval(self, z):
        L = self.L_inf*self.a*(z/self.L_inf)**self.nu
        return L**2

###################################################################################################

class MaternExpansion(BasicExpansion):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nPars = 2
        self.eval_vec = np.vectorize(self.eval)

    def __call__(self, z):
        return self.eval_vec(z)

    def update(self, params):
        params_iter = iter(params)
        self.a  = np.abs(next(params_iter))
        self.nu = np.abs(next(params_iter))

    def eval(self, z):
        L = self.L_inf*(1 - Mv(self.nu, self.a*z/self.L_inf))
        return L**2

###################################################################################################




###################################################################################################
###################################################################################################

if __name__ == "__main__":

    '''TEST by method of manufactured solutions'''

    pass