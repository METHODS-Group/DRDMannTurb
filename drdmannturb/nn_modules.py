"""
This module contains all the implementations PyTorch nn.Module subclasses
used throughout.
"""

__all__ = ["TauResNet", "TauNet", "CustomNet"]

from typing import Any, Callable, List, Union

import torch
import torch.nn as nn

"""
Learnable functions
"""


class Rational(nn.Module):
    """
    Learnable rational kernel
    """

    def __init__(self, nModes: int, learn_nu: bool = True) -> None:
        """
        Constructor for Rational

        Parameters
        ----------
        learn_nu : bool, optional
            Indicates whether or not the exponent nu should be learned
            also; by default True
        """
        super().__init__()
        self.nModes = nModes
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
            [nn.Linear(hlayer, hlayer, bias=False).double() for _ in range(nlayers - 1)]
        )
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

        num_layers = len(hlayers)
        self.linears = nn.ModuleList(
            [
                nn.Linear(hlayers[k], hlayers[k + 1], bias=False).double()
                for k in range(num_layers - 1)
            ]
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

        for idx, lin in enumerate(self.linears):
            out = self.activations[idx](lin(out))

        out = self.linear_out(out)

        return x + out


"""
Learnable eddy lifetime models
"""


class ResNetBlock(nn.Module):
    def __init__(self, inlayer=3, outlayer=3):
        super(ResNetBlock, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(inlayer, outlayer, bias=False).double(),
            # nn.LayerNorm(outlayer),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(outlayer, outlayer, bias=False).double(),
            # nn.LayerNorm(outlayer),
            # nn.ReLU()
        )

        self.outlayer = outlayer
        self.relu = nn.ReLU()

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        residual = x

        output = self.fc1(x)

        output = self.fc2(output)

        output += residual
        output = self.relu(output)
        # output = nn.ReLU()(output)

        return output

    def forward(self, x):
        return self._forward_impl(x)


class ResNet(nn.Module):
    """
    ResNet implementation
    """

    def __init__(
        self, n_layers: list[int], inlayer: int = 3, outlayer: int = 3
    ) -> None:
        super(ResNet, self).__init__()
        self.indims = 10  # not of the data but after the first layer upward

        # this serves as a substitute for the initial conv
        # present in resnets for image-based tasks
        self.layer0 = nn.Sequential(
            nn.Linear(inlayer, self.indims, bias=False).double(), nn.ReLU()
        )

        # TODO: need to downsample if not 4...??????
        self.block1 = self._make_layer(n_layers[0], self.indims)
        self.block2 = self._make_layer(n_layers[1], self.indims)

        self.fc = nn.Linear(self.indims, outlayer).double()

    def _make_layer(self, blocks, indims):
        layers = []
        layers.append(ResNetBlock(self.indims, indims))

        self.indims = indims

        for _ in range(1, blocks):
            layers.append(ResNetBlock(inlayer=self.indims, outlayer=indims))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.layer0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

##############################################################################
# Below here are exposed.
##############################################################################


class TauResNet(nn.Module):
    """
    tauResNet implementation

    Consists of ResNet and Rational
    """

    def __init__(
        self,
        hidden_layer_sizes: List[int] = [10, 10],
        n_modes: int = 10,
        learn_nu: bool = True,
    ):
        """
        Constructor for res-net implementation of the learnable eddy lifetime model

        Parameters
        ----------
        hidden_layer_sizes : List[int], optional
            List of integers greater than zero each describing the size of the
            respectively indexed , by default [10, 10]
        n_modes : int, optional
            _description_, by default 10
        learn_nu : bool, optional
            _description_, by default True
        """
        super(TauResNet, self).__init__()

        self.hlayers = hidden_layer_sizes

        self.n_modes = n_modes
        self.fg_learn_nu = learn_nu

        # TODO: change activations list here and propagate through to resnet blocks
        self.NN = ResNet(n_layers=self.hlayers, inlayer=3, outlayer=3)
        self.Ra = Rational(nModes=self.n_modes, learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(
        self, f: Callable[[torch.Tensor], torch.Tensor], k: torch.Tensor
    ) -> torch.Tensor:
        """
        TODO -- what exactly?

        Parameters
        ----------
        f : Callable[[torch.Tensor], torch.Tensor]
            A function that takes a tensor and returns a tensor
        k : torch.Tensor
            _description_

            # TODO -- wave numbers?

        Returns
        -------
        torch.Tensor
            # TODO -- what???
        """
        return 0.5 * (f(k) + f(k * self.sign))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation

        Parameters
        ----------
        k : torch.Tensor
            TODO -- greater meaning... wave numbers?

        Returns
        -------
        torch.Tensor
            TODO -- greater meaning ???
        """
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)

        return tau


class TauNet(nn.Module):
    """
    TauNet implementation

    Consists of a SimpleNN and a Rational
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_layer_size: int = 3,
        n_modes: int = 10,
        learn_nu: bool = True,
    ):
        """
        Constructor for TauNet

        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        hidden_layer_size : int, optional
            Size of the hidden layers, by default 3
        n_modes : int, optional
            Number of wave modes, by default 10
        learn_nu : bool, optional
            If true, learns also the exponent Nu, by default True
        """

        super(TauNet, self).__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.n_modes = n_modes
        self.fg_learn_nu = learn_nu

        self.NN = SimpleNN(
            nlayers=self.n_layers, inlayer=3, hlayer=self.hidden_layer_size, outlayer=3
        )
        self.Ra = Rational(nModes=self.n_modes, learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(self, f: Callable, k: torch.Tensor) -> torch.Tensor:
        """
        TODO -- what is this?

        Parameters
        ----------
        f : Callable
            _description_
        k : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        return 0.5 * (f(k) + f(k * self.sign))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation.

        Parameters
        ----------
        k : torch.Tensor
            Wave vector input

        Returns
        -------
        torch.Tensor
            _description_
        """
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)
        return tau


class CustomNet(nn.Module):
    """
    CustomNet implementation. Consists of a CustomMLP and
    Rational module.
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_layer_sizes: Union[int, list[int]] = [10, 10],
        activations: List[Callable] = [nn.ReLU(), nn.ReLU()],
        n_modes: int = 10,
        learn_nu: bool = True,
    ):
        """
        Constructor for the customNet

        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        activations : List[Any], optional
            List of activation functions to use, by default [nn.ReLU(), nn.ReLU()]

            TODO -- type hint this properly
        n_modes : int, optional
            Number of wave modes, by default 10
        learn_nu : bool, optional
            Determines whether or not the exponent Nu is also learned, by default True
        """

        super().__init__()

        self.n_layers = n_layers
        self.activations = activations

        self.n_modes = n_modes
        self.fg_learn_nu = learn_nu

        hls = None
        if type(hidden_layer_sizes) is int:
            hls = [hidden_layer_sizes for _ in range(n_layers)]
        else:
            hls = hidden_layer_sizes

        self.NN = CustomMLP(
            hlayers=hls, activations=self.activations, inlayer=3, outlayer=3
        )
        self.Ra = Rational(nModes=self.n_modes, learn_nu=self.fg_learn_nu)

        self.sign = torch.tensor([1, -1, 1], dtype=torch.float64).detach()

    def sym(
        self, f: Callable[[torch.Tensor], torch.Tensor], k: torch.Tensor
    ) -> torch.Tensor:
        """
        TODO -- figure out what exactly this is

        Parameters
        ----------
        f : Callable[[torch.Tensor], torch.Tensor]
            A function that takes a Tensor and produces a Tensor
        k : torch.Tensor
            TODO -- greater meaning

        Returns
        -------
        torch.Tensor
            TODO -- greater meaning
        """
        return 0.5 * (f(k) + f(k * self.sign))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation

        Parameters
        ----------
        k : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        k_mod = self.NN(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)
        return tau
