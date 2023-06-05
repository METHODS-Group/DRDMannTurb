
import torch
import torch.nn as nn

from .LearnableFunctions import Rational, SimpleNN

"""
==================================================================================================================
Learnable Eddy Liftime class
==================================================================================================================
"""

class tauNet(nn.Module):
    def __init__(self, **kwargs):
        super(tauNet, self).__init__()        
        self.nlayers = kwargs.get('nlayers', 2)
        self.hidden_layer_size = kwargs.get('hidden_layer_size', 3)
        self.nModes            = kwargs.get('nModes', 10)
        self.fg_learn_nu       = kwargs.get('learn_nu', True)

        self.NN = SimpleNN(nlayers=self.nlayers, inlayer=3, hlayer=self.hidden_layer_size, outlayer=3)
        self.Ra = Rational(nModes=self.nModes, learn_nu=self.fg_learn_nu)

        # self.T = nn.Linear(3,3,bias=False).double()
        

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(self, f, k):
        return 0.5*(f(k) + f(k*self.sign))


    def forward(self, k):
        # NN    = self.sym(self.NN, k)
        # NN    = self.NN(k**2)
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau   = self.Ra(k_mod)
        return tau

        # k2 = k.norm(dim=-1,keepdim=True).square()
        # l = k/k2
        # tau = self.NN(l) + self.NN(l*self.sign)
        # # k_mod = k.norm(dim=-1) + NN.squeeze(-1)
        # # tau   = self.Ra(k_mod)
        # return tau.squeeze(-1)

        # k_mod = self.T(k).norm(dim=-1)
        # tau   = self.Ra(k_mod)
        # return tau