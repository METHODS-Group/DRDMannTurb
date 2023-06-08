
from math import *
from pylab import *
import numpy as np
import scipy.optimize
import scipy.fftpack as fft
from scipy.special import gamma
import copy
from time import time
import matplotlib.pyplot as plt 


###################################################################################################
#  Inference of design parameters
###################################################################################################

class ModelParametersInference:

    def __init__(self, ModelDescriptor, loc, Data, weights=None, jac=False, verbose=False):
        self.verbose = verbose

        self.Model = ModelDescriptor
        self.loc = loc
        self.Data = Data
        self.weights = weights
        self.jac = jac
        
        # self.ModelParameters = self.infer()


    ### Objective function
    def Objective(self, p):
        if not self.jac:
            model = self.Model(p, self.loc)
        else:
            model, grad_model = self.Model(p, self.loc, self.jac)
        # misfit = np.log(model) - np.log(self.Data)
        misfit = model - self.Data
        # misfit = misfit[:,:misfit.shape[1]//2,:]
        # misfit = misfit[(0,2),:]
        # misfit = misfit[0 ,:,:]
        if self.weights:
            misfit *= np.sqrt(self.weights)
        J = 0.5*np.sum(misfit**2)

        # if (self.Model.ExpansionType is 'Exp') and self.Model.Expansion.nTerms==1:
        #     E = self.Model.Expansion
        #     g = 100
        #     J += g * E.c[0]**2/(2*E.d[0]) * np.exp(-2*E.d[0])
        # elif (self.Model.ExpansionType is 'MP') and self.Model.Expansion.nTerms==1:
        #     E = self.Model.Expansion
        #     g = 100
        #     J += g * E.c[0]**2/(1+E.d[0]**2)
        # if (self.Model.ExpansionType is 'MP') and self.Model.Expansion.nTerms==1:
        #     c, d, L = self.Model.Expansion.c[0], self.Model.Expansion.d[0], self.Model.Expansion.L_inf
        #     g = 1.e3
        #     J += g * c**2/(2+d**2/L)

        self.iter += 1
        if self.verbose:
            print('\nFunction call {0:d}: residuum = {1}'.format(self.iter, J))
            print('parameters = {0}'.format(p))

        if not self.jac:
            return J
        else:
            misfit = misfit.flatten() 
            grad_model = grad_model.reshape([-1,self.Model.nPar])
            JacJ = misfit @ grad_model
            if (self.Model.ExpansionType is 'Exp') and self.Model.Expansion.nTerms==1:
                extraJacJ = g * E.c[0]/E.d[0] * np.array([ 1, -E.c[0]*(1/(2*E.d[0]) + 1), 0 ]) * np.exp(-2*E.d[0])
                JacJ += extraJacJ
            # self.jac = False
            # func = lambda p: self.Model(p, self.loc)[0]
            # JacAppx = scipy.optimize.approx_fprime(p, self.Objective, 1.e-6)
            # self.jac = True
            # print('JAC ERR: ', np.linalg.norm(JacAppx-JacJ) )
            return J, JacJ


    ### Estimator
    def infer(self, p_ini=None, **kwargs):
        if p_ini is None:
            p_ini = kwargs.get('init_guess', self.Model.default_parameters())
        tol = kwargs.get('tol', 1.e-3)
        self.jac = kwargs.get('jac', self.jac)
        J = lambda p: self.Objective(p)
        self.iter = 0
        result = scipy.optimize.minimize(J, p_ini,jac=self.jac,tol=tol,options={'disp': self.verbose})
        p_opt = result.x
        return p_opt








###################################################################################################
###################################################################################################

if __name__ == "__main__":

    '''TEST by method of manufactured solutions'''

    pass