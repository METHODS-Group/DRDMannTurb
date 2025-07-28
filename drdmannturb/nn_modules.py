"""All nn.Module's used throughout the framework."""

__all__ = ["TauNet"]


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

    fg_learn_nu: bool
    nu: torch.Tensor | nn.Parameter

    def __init__(self, learn_nu: bool = True, nu_init: float = -1.0 / 3.0) -> None:
        """
        Initialize the rational kernel.

        Parameters
        ----------
        learn_nu : bool, optional
            Indicates whether or not the exponent nu should be learned
            also; by default True
        nu_init : float, optional
            Initial value for the nu parameter; by default -1.0/3.0
        """
        super().__init__()
        self.fg_learn_nu = learn_nu

        self.nu = torch.tensor(nu_init)
        if self.fg_learn_nu:
            self.nu = nn.Parameter(self.nu)

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

        # Handle numerical instability for very large or very small values
        # For very large out, use asymptotic behavior
        large_mask = out > 1e6
        small_mask = out < 1e-6

        result = torch.zeros_like(out)

        # For normal range
        normal_mask = ~(large_mask | small_mask)
        if normal_mask.any():
            out_normal = out[normal_mask]
            result[normal_mask] = (out_normal**a) / ((1 + out_normal**2) ** b)

        # For very large values, use asymptotic behavior: out^(a-2b)
        if large_mask.any():
            result[large_mask] = out[large_mask]**(a - 2*b)

        # For very small values, use Taylor expansion around 0
        if small_mask.any():
            result[small_mask] = out[small_mask]**a

        return result


class TauNet(nn.Module):
    r"""
    A neural network designed for approximating the eddy lifetime function :math:`\tau(\boldsymbol{k})`.

    By default, this composes a multi-layer perceptron with a "rational kernel," which is used
    to enforce certain analytic behaviors.

    The objective is to learn the function

    .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k}),

    where

    .. math::
        \boldsymbol{a}(\boldsymbol{k}) = \operatorname{abs}(\boldsymbol{k}) +
        \mathrm{NN}(\operatorname{abs}(\boldsymbol{k})).
    """

    def __init__(
        self,
        n_layers: int = 2,
        hidden_layer_sizes: int | list[int] = 10,
        activations: list[nn.Module] | nn.Module | None = None,
        learn_nu: bool = True,
        nu_init: float = -1.0 / 3.0,
    ):
        r"""
        Initialize the TauNet.

        Parameters
        ----------
        n_layers : int, optional
            Number of hidden layers, by default 2
        hidden_layer_sizes : Union[int, list[int]], optional
            Sizes of each layer; by default 10
        activations : Union[list[nn.Module], nn.Module, None], optional
            Activation functions to use, by default None (uses ReLU)
        learn_nu : bool, optional
            If true, learns also the exponent nu, by default True
        nu_init : float, optional
            Initial value for the nu parameter, by default -1.0/3.0
        """
        super().__init__()

        # Validate n_layers
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")
        self.n_layers = n_layers

        # Handle layer sizes
        if isinstance(hidden_layer_sizes, int):
            if hidden_layer_sizes <= 0:
                raise ValueError("hidden_layer_sizes must be a positive integer")
            hidden_layer_sizes = [hidden_layer_sizes] * n_layers
        else:
            if len(hidden_layer_sizes) != n_layers:
                raise ValueError("hidden_layer_sizes must be a list of integers of length n_layers")
            for size in hidden_layer_sizes:
                if size <= 0:
                    raise ValueError("hidden_layer_sizes must be a list of positive integers")

        self.hidden_layer_sizes = hidden_layer_sizes

        # Handle activations
        if activations is None:
            self.activations = [nn.ReLU() for _ in range(n_layers)]
        elif isinstance(activations, nn.Module):
            self.activations = [activations] * n_layers
        elif isinstance(activations, list):
            if len(activations) != n_layers:
                raise ValueError("activations must be a list of nn.Module's of length n_layers")
            for act in activations:
                if not isinstance(act, nn.Module):
                    raise ValueError("activations must be a list of nn.Module's")
            self.activations = activations

        # Build MLP layers - includes input, hidden, and output layers
        layers = [nn.Linear(3, hidden_layer_sizes[0], bias=False)]  # Input layer

        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1], bias=False))

        # Output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], 3, bias=False))

        self.linears = nn.ModuleList(layers)

        # Rational kernel
        self.Ra = Rational(learn_nu=learn_nu, nu_init=nu_init)

        # Initialize with small noise to prevent zero initialization
        noise_magnitude = 1.0e-8
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn(param.size()) * noise_magnitude)

        self.sign = torch.tensor([1, -1, 1]).detach()

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layers."""
        out = x.clone()

        # Apply all layers except the last one with activation
        for idx, layer in enumerate(self.linears[:-1]):
            out = self.activations[idx](layer(out))

        # Apply final layer without activation
        out = self.linears[-1](out)

        return x + out  # Residual connection

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the TauNet.

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
        # Debug: Check input (just summary stats)
        if torch.isnan(k).any():
            print(f"NaN in TauNet input k: min={k.min().item()}, max={k.max().item()}, mean={k.mean().item()}")

        k_mod = self._mlp_forward(k.abs()).norm(dim=-1)

        # Debug: Check k_mod (just summary stats)
        if torch.isnan(k_mod).any():
            print(f"NaN in TauNet k_mod: min={k_mod.min().item()}, max={k_mod.max().item()}, mean={k_mod.mean().item()}")
            mlp_out = self._mlp_forward(k.abs())
            print(f"MLP output: min={mlp_out.min().item()}, max={mlp_out.max().item()}, mean={mlp_out.mean().item()}")

        tau = self.Ra(k_mod)

        # Debug: Check tau (just summary stats)
        if torch.isnan(tau).any():
            print(f"NaN in TauNet tau: min={tau.min().item()}, max={tau.max().item()}, mean={tau.mean().item()}")

        return tau
