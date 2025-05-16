"""
This module contains all the implementations PyTorch nn.Module subclasses
used throughout.
"""

__all__ = ["TauNet", "CustomNet"]

from typing import List, Union

import torch
import torch.nn as nn


class Rational(nn.Module):
    r"""
    Learnable rational kernel; a neural network that learns the rational function

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}},
            \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}),

        specifically, the neural network part of the augmented wavevector

        .. math::
            \mathrm{NN}(\operatorname{abs}(\boldsymbol{k})).
    """

    def __init__(self, learn_nu: bool = True) -> None:
        """
        Parameters
        ----------
        learn_nu : bool, optional
            Indicates whether or not the exponent nu should be learned
            also; by default True
        """
        super().__init__()
        self.fg_learn_nu = learn_nu
        self.nu = -1.0 / 3.0
        if self.fg_learn_nu:
            self.nu = nn.Parameter(torch.tensor(float(self.nu)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation

        Parameters
        ----------
        x : torch.Tensor
            Network input

        Returns
        -------
        torch.Tensor
            Network output
        """
        a = self.nu - 2 / 3
        b = self.nu
        out = torch.abs(x)
        out = out**a / (1 + out**2) ** (b / 2)
        return out


class SimpleNN(nn.Module):
    """
    A simple feed-forward neural network consisting of n layers with a ReLU activation function. The default
    initialization is to random noise of magnitude 1e-9.
    """

    def __init__(self, nlayers: int = 2, inlayer: int = 3, hlayer: int = 3, outlayer: int = 3) -> None:
        """
        Parameters
        ----------
        nlayers : int, optional
            Number of layers to use, by default 2
        inlayer : int, optional
            Number of input features, by default 3
        hlayer : int, optional
            Number of hidden layers, by default 3
        outlayer : int, optional
            Number of output features, by default 3
        """
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(hlayer, hlayer, bias=False).double() for _ in range(nlayers - 1)])
        self.linears.insert(0, nn.Linear(inlayer, hlayer, bias=False).double())
        self.linear_out = nn.Linear(hlayer, outlayer, bias=False).double()

        self.actfc = nn.ReLU()

        # NOTE: init parameters with noise
        noise_magnitude = 1.0e-9
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation

        Parameters
        ----------
        x : torch.Tensor
            Network input

        Returns
        -------
        torch.Tensor
            Network output
        """
        out = x.clone()

        for lin in self.linears:
            out = self.actfc(lin(out))

        out = self.linear_out(out)

        return out + x


"""
Learnable eddy lifetime models
"""

##############################################################################
# Below here are exposed.
##############################################################################


class TauNet(nn.Module):
    r"""
    Classical implementation of neural network that learns the eddy lifetime function :math:`\tau(\boldsymbol{k})`.
    A SimpleNN and Rational network comprise this class. The network widths are determined by a single integer and
    thereafter the networks have hidden layers of only that width.

    The objective is to learn the function

    .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}),

    where

    .. math::
        \boldsymbol{a}(\boldsymbol{k}) = \operatorname{abs}(\boldsymbol{k}) +
        \mathrm{NN}(\operatorname{abs}(\boldsymbol{k})).

    This class implements the simplest architectures which solve this problem.
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_layer_size: int = 3,
        learn_nu: bool = True,
    ):
        r"""
        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        hidden_layer_size : int, optional
            Size of the hidden layers, by default 3
        learn_nu : bool, optional
            If true, learns also the exponent :math:`\nu`, by default True
        """
        super(TauNet, self).__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.fg_learn_nu = learn_nu

        self.NN = SimpleNN(nlayers=self.n_layers, inlayer=3, hlayer=self.hidden_layer_size, outlayer=3)
        self.Ra = Rational(learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation. Evaluates

        Parameters
        ----------
        k : torch.Tensor
            Wave vector input

        Returns
        -------
        torch.Tensor
            Output of forward pass of neural network.
        """
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)
        return tau

