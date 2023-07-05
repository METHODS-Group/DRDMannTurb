
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

"""
==================================================================================================================
Learnable rational kernel
==================================================================================================================
"""

# class Rational(nn.Module):
#     def __init__(self, nModes=20):
#         super().__init__()
#         self.nModes  = nModes
#         # self.poles   = torch.logspace(-3, 3, self.nModes, dtype=torch.float64)**(1/2)
#         # # self.poles   = torch.cat((torch.tensor([0], dtype=torch.float64), self.poles))
#         # # self.poles   = torch.linspace(0, 1000, self.nModes, dtype=torch.float64)**(1/2)
#         # self.weights = 1.e-3*torch.rand((self.nModes,), dtype=torch.float64)

#         # self.poles   = nn.Parameter(self.poles)
#         # self.weights = nn.Parameter(self.weights)


#         self.zeros  = nn.Parameter(1. + torch.arange(self.nModes))
#         self.poles  = nn.Parameter(1. + torch.arange(self.nModes))
#         self.factor = nn.Parameter(torch.tensor(1.))

#         ### init parameters with noise
#         noise_magnitude = 1.e-1
#         with torch.no_grad():
#             for param in self.parameters():
#                 param.add_(torch.randn(param.size()) * noise_magnitude)

#         self.factor.data = torch.tensor(1.e-1)



#     # def forward(self, x):
#     #     den = x.unsqueeze(-1) + self.poles**2
#     #     out = self.weights**2 / den
#     #     out = out.sum(dim=-1)
#     #     return out

#     def forward(self, x):
#         p = x.unsqueeze(-1) + self.zeros**2
#         q = x.unsqueeze(-1) + self.poles**2
#         out = self.factor**2 * (p/q).prod(dim=-1)
#         return out


class Rational(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.fg_learn_nu = kwargs.get('learn_nu', True)
        self.nu = -1./3.
        if self.fg_learn_nu:
            self.nu = nn.Parameter(torch.tensor(float(self.nu)))
        # self.scale = nn.Parameter(torch.tensor(1.))
        # self.coef  = nn.Parameter(torch.tensor(1.))

        # self.alpha = nn.Parameter(-1.*torch.arange(1,4))
        # # self.scale = nn.Parameter(torch.tensor(1.))
        # self.coef  = nn.Parameter(torch.ones_like(self.alpha))


    def forward(self, x):
        a = self.nu - 2/3
        b = self.nu
        out = torch.abs(x)
        out = out**a / (1 + out**2)**(b/2)
        return out #* self.scale**2

    # def forward(self, x):
    #     out = self.coef**2 * torch.abs(x).unsqueeze(-1)**self.alpha
    #     out = out.sum(dim=-1)
    #     print('coefs = ', parameters_to_vector(self.coef))
    #     print('alpha = ', parameters_to_vector(self.alpha))
    #     return out #* self.scale**2




"""
==================================================================================================================
Simple Neural Net
==================================================================================================================
"""

class SimpleNN(nn.Module):
    def __init__(self, nlayers=2, inlayer=3, hlayer=3, outlayer=3):
        super().__init__()
        self.linears = nn.ModuleList([ nn.Linear(hlayer, hlayer, bias=False).double() for l in range(nlayers-1) ])
        self.linears.insert(0, nn.Linear(inlayer, hlayer, bias=False).double())
        self.linear_out = nn.Linear(hlayer, outlayer, bias=False).double()
        # self.actfc = nn.Softplus()
        self.actfc = nn.ReLU()

        ### init parameters with noise
        noise_magnitude = 1.e-9
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

        # self.powers = torch.arange(1,4).detach()

    def forward(self, x):
        out = x.clone()
        # out = (out.unsqueeze(-1)**self.powers).flatten(start_dim=-2, end_dim=-1)
        for lin in self.linears: out = self.actfc(lin(out))
        out = self.linear_out(out)
        # out = out**2
        # out = self.actfc(out)
        # out = 1 + torch.tanh(out)
        out = x + out
        # out = out.norm(dim=-1)
        return out


class CustomMLP(nn.Module):
    def __init__(self, hlayers, activations, inlayer=3, outlayer=3) -> None:
        """_summary_

        Parameters
        ----------
        hlayers : list
            list specifying widths of hidden layers in NN
        activations : _type_
            list specifying activation functions for each hidden layer
        inlayer : int, optional
            _description_, by default 3
        outlayer : int, optional
            _description_, by default 3
        """
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(hlayer, hlayer, bias=False).double() for hlayer in hlayers])
        self.linears.insert(0, nn.Linear(inlayer, hlayers[0], bias=False).double())
        self.linear_out = nn.Linear(hlayers[-1], outlayer, bias=False).double()

        self.activations = activations

        noise_magnitude = 1.e-9 
        with torch.no_grad(): 
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

    def forward(self, x):
        out = x.clone()

        for lin, activ in zip(self.linears, self.activations): out = activ(lin(out))

        out = self.linear_out(out)

        return x + out
