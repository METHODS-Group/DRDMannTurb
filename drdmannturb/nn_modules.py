"""All nn.Module's used throughout the framework."""

__all__ = ["TauNet", "CustomNet"]

from typing import List, Union

import torch
import torch.nn as nn


class Rational(nn.Module):
    r"""
    Learnable rational kernel.

    We require the MLP to be composed with this kernel to ensure certain
    properties and asymptotics of the eddy lifetime function in the end.

        .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}},
            \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}),

        specifically, the neural network part of the augmented wavevector

        .. math::
            \mathrm{NN}(\operatorname{abs}(\boldsymbol{k})).
    """

    def __init__(self, learn_nu: bool = True, k_inf_asymptote: float = -2.0 / 3.0) -> None:
        """
        Initialize the rational kernel.

        Parameters
        ----------
        learn_nu : bool, optional
            Indicates whether or not the exponent nu should be learned
            also; by default True
        """
        super().__init__()
        self.fg_learn_nu = learn_nu

        self.k_inf_asymptote = k_inf_asymptote

        self.nu = -1.0 / 3.0
        if self.fg_learn_nu:
            self.nu = nn.Parameter(torch.tensor(float(-1.0 / 3.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the rational kernel.

        Parameters
        ----------
        x : torch.Tensor
            Network input

        Returns
        -------
        torch.Tensor
            Network output
        """
        a = self.nu - self.k_inf_asymptote
        b = self.nu / 2.0
        out = torch.abs(x)
        out = (out**a) / ((1 + out**2) ** b)
        return out


class SimpleNN(nn.Module):
    """
    A simple feed-forward neural network consisting of n layers with a ReLU activation function.

    The default initialization is to random noise of magnitude 1e-9.
    """

    def __init__(self, nlayers: int = 2, inlayer: int = 3, hlayer: int = 3, outlayer: int = 3) -> None:
        """
        Initialize a simple feed-forward network.

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
        Forward pass of the simple feed-forward network.

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


class CustomMLP(nn.Module):
    """
    Feed-forward neural network with variable widths of layers and activation functions.

    Useful for DNN configurations and experimentation with different activation functions.
    """

    def __init__(
        self,
        hlayers: list[int],
        activations: list[nn.Module],
        inlayer: int = 3,
        outlayer: int = 3,
    ) -> None:
        """
        Initialize a feed-forward network with variable widths of layers and activation functions.

        Parameters
        ----------
        hlayers : list
            list specifying widths of hidden layers in NN
        activations : list[nn.Module]
            list specifying activation functions for each hidden layer
        inlayer : int, optional
            Number of input features, by default 3
        outlayer : int, optional
            Number of features to output, by default 3
        """
        super().__init__()

        num_layers = len(hlayers)
        self.linears = nn.ModuleList(
            [nn.Linear(hlayers[k], hlayers[k + 1], bias=False).double() for k in range(num_layers - 1)]
        )
        self.linears.insert(0, nn.Linear(inlayer, hlayers[0], bias=False).double())
        self.linear_out = nn.Linear(hlayers[-1], outlayer, bias=False).double()

        self.activations = activations

        noise_magnitude = 1.0e-9
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network with variable widths of layers and activation functions.

        Parameters
        ----------
        x : torch.Tensor
            Input to the network

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        out = x.clone()

        for idx, lin in enumerate(self.linears):
            out = self.activations[idx](lin(out))

        out = self.linear_out(out)

        return x + out


"""
Learnable eddy lifetime models
"""

##############################################################################
# Below here are exposed.
##############################################################################


class TauNet(nn.Module):
    r"""
    A neural network which learns the eddy lifetime function :math:`\tau(\boldsymbol{k})`.

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
        k_inf_asymptote: float = -2.0 / 3.0,
    ):
        r"""
        Initialize the tauNet.

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
        self.Ra = Rational(learn_nu=self.fg_learn_nu, k_inf_asymptote=k_inf_asymptote)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the tauNet.

        Evaluates the eddy lifetime function :math:`\tau(\boldsymbol{k})`.

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


class CustomNet(nn.Module):
    r"""
    A more versatile version of the tauNet.

    The objective is the same: to learn the eddy lifetime function
    :math:`\tau(\boldsymbol{k})` in the same way. This class allows for neural networks of variable widths and
    different kinds of activation functions used between layers.
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_layer_sizes: Union[int, list[int]] = [10, 10],
        activations: List[nn.Module] = [nn.ReLU(), nn.ReLU()],
        learn_nu: bool = True,
        k_inf_asymptote: float = -2.0 / 3.0,
    ):
        r"""
        Initialize the customNet.

        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        hidden_layer_sizes : Union[int, list[int]]
            Sizes of each layer; by default [10, 10].
        activations : List[nn.Module], optional
            List of activation functions to use, by default [nn.ReLU(), nn.ReLU()]
        learn_nu : bool, optional
            If true, learns also the exponent :math:`\nu`, by default True
        """
        super().__init__()

        self.n_layers = n_layers
        self.activations = activations

        self.fg_learn_nu = learn_nu

        hls: List[int]
        if isinstance(hidden_layer_sizes, int):
            hls = [hidden_layer_sizes for _ in range(n_layers)]
        else:
            hls = hidden_layer_sizes

        self.NN = CustomMLP(hlayers=hls, activations=self.activations, inlayer=3, outlayer=3)
        self.Ra = Rational(learn_nu=self.fg_learn_nu, k_inf_asymptote=k_inf_asymptote)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the customNet.

        Evaluates the eddy lifetime function :math:`\tau(\boldsymbol{k})`.

        Parameters
        ----------
        k : torch.Tensor
            Input wavevector domain.

        Returns
        -------
        torch.Tensor
            Output of forward pass of neural network
        """
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)
        return tau
