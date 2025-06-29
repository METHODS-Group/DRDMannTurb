"""All nn.Module's used throughout the framework."""

__all__ = ["TauNet", "CustomNet"]


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

    def __init__(self, learn_nu: bool = True, nu_init: float = -1.0 / 3.0) -> None:
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

        self.nu = nu_init
        if self.fg_learn_nu:
            self.nu = nn.Parameter(torch.tensor(float(nu_init)))

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
        a = self.nu - (2.0 / 3.0)
        b = self.nu / 2.0
        out = torch.abs(x)
        out = (out**a) / ((1 + out**2) ** b)
        return out


class TauNet(nn.Module):
    r"""
    A neural network which learns the eddy lifetime function :math:`\tau(\boldsymbol{k})`.

    This network combines a simple feed-forward MLP with a rational kernel. The network widths are
    determined by a single integer and thereafter the networks have hidden layers of only that width.

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
        self, n_layers: int = 2, hidden_layer_size: int = 3, learn_nu: bool = True, nu_init: float = -1.0 / 3.0
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
        super().__init__()

        self.n_layers = n_layers
        self.hidden_layer_size = hidden_layer_size
        self.fg_learn_nu = learn_nu

        # Build MLP layers directly (formerly SimpleNN functionality)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_layer_size, hidden_layer_size, bias=False) for _ in range(n_layers - 1)]
        )
        self.linears.insert(0, nn.Linear(3, hidden_layer_size, bias=False))  # inlayer = 3
        self.linear_out = nn.Linear(hidden_layer_size, 3, bias=False)  # outlayer = 3

        self.actfc = nn.ReLU()

        # Rational kernel
        self.Ra = Rational(learn_nu=self.fg_learn_nu, nu_init=nu_init)

        # Initialize with small noise to prevent zero initialization
        noise_magnitude = 1.0e-9
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

        self.sign = torch.tensor([1, -1, 1]).detach()

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layers."""
        out = x.clone()

        for lin in self.linears:
            out = self.actfc(lin(out))

        out = self.linear_out(out)
        return out + x  # Residual connection

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
        k_mod = self._mlp_forward(k.abs()).norm(dim=-1)
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
        hidden_layer_sizes: int | list[int] = [10, 10],
        activations: list[nn.Module] | None = None,
        learn_nu: bool = True,
        nu_init: float = -1.0 / 3.0,
    ):
        r"""
        Initialize the customNet.

        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        hidden_layer_sizes : Union[int, list[int]]
            Sizes of each layer; by default [10, 10].
        activations : list[nn.Module], optional
            List of activation functions to use, by default [nn.ReLU(), nn.ReLU()]
        learn_nu : bool, optional
            If true, learns also the exponent :math:`\nu`, by default True
        """
        super().__init__()

        self.n_layers = n_layers
        self.fg_learn_nu = learn_nu

        # Handle default activations
        if activations is None:
            activations = [nn.ReLU() for _ in range(n_layers)]
        self.activations = activations

        # Handle layer sizes
        hls: list[int]
        if isinstance(hidden_layer_sizes, int):
            hls = [hidden_layer_sizes for _ in range(n_layers)]
        else:
            hls = hidden_layer_sizes

        # Build MLP layers directly (formerly CustomMLP functionality)
        num_layers = len(hls)
        self.linears = nn.ModuleList([nn.Linear(hls[k], hls[k + 1], bias=False) for k in range(num_layers - 1)])
        self.linears.insert(0, nn.Linear(3, hls[0], bias=False))  # inlayer = 3
        self.linear_out = nn.Linear(hls[-1], 3, bias=False)  # outlayer = 3

        # Rational kernel
        self.Ra = Rational(learn_nu=self.fg_learn_nu, nu_init=nu_init)

        # Initialize with small noise to prevent zero initialization
        noise_magnitude = 1.0e-9
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

        self.sign = torch.tensor([1, -1, 1]).detach()

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the custom MLP layers."""
        out = x.clone()

        for idx, lin in enumerate(self.linears):
            out = self.activations[idx](lin(out))

        out = self.linear_out(out)
        return x + out  # Residual connection

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
        k_mod = self._mlp_forward(k.abs()).norm(dim=-1)
        tau = self.Ra(k_mod)
        return tau
