from typing import Callable

import torch
import torch.nn as nn


class Rational(nn.Module):
    """
    Learnable rational kernel
    """

    def __init__(self, learn_nu: bool = True) -> None:
        """
        Constructor for Rational

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
        return out  # * self.scale**2


class SimpleNN(nn.Module):
    def __init__(
        self, nlayers: int = 2, inlayer: int = 3, hlayer: int = 3, outlayer: int = 3
    ) -> None:
        """
        Constructor for SimpleNN

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
        self.linears = nn.ModuleList(
            [nn.Linear(hlayer, hlayer, bias=False).double() for l in range(nlayers - 1)]
        )
        self.linears.insert(0, nn.Linear(inlayer, hlayer, bias=False).double())
        self.linear_out = nn.Linear(hlayer, outlayer, bias=False).double()

        # self.actfc = nn.Softplus()
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
        # out = (out.unsqueeze(-1)**self.powers).flatten(start_dim=-2, end_dim=-1)
        for lin in self.linears:
            out = self.actfc(lin(out))
        out = self.linear_out(out)
        # out = out**2
        # out = self.actfc(out)
        # out = 1 + torch.tanh(out)
        out = x + out
        # out = out.norm(dim=-1)
        return out


class CustomMLP(nn.Module):
    def __init__(
        self,
        hlayers: list[int],
        activations: list[Callable],
        inlayer: int = 3,
        outlayer: int = 3,
    ) -> None:
        """
        Constructor for CustomMLP

        Parameters
        ----------
        hlayers : list
            list specifying widths of hidden layers in NN
        activations : _type_
            list specifying activation functions for each hidden layer
        inlayer : int, optional
            Number of input features, by default 3
        outlayer : int, optional
            Number of features to output, by default 3
        """
        super().__init__()

        self.linears = nn.ModuleList(
            [nn.Linear(hlayer, hlayer, bias=False).double() for hlayer in hlayers]
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
        Forward method implementation

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

        for lin, activ in zip(self.linears, self.activations):
            out = activ(lin(out))

        out = self.linear_out(out)

        return x + out
