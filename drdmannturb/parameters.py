"""Several dataclasses that make it easy to pass around parameters."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .enums import EddyLifetimeType

__all__ = ["ProblemParameters", "PhysicalParameters", "NNParameters", "LossParameters"]


@dataclass
class ProblemParameters:
    r"""Define generic numerical parameters for the problem.

    This class provides a convenient method of storing and passing around
    generic numerical parameters; this also offers default values

    Args
    ----
    num_components : int
        Number of components to fit, either 3, 4, or 6. By default, 4.
        - If 3, assumes that 11, 22, and 33 are provided in that order.
        - If 4, assumes that 11, 22, 33, and 13 are provided in that order.
        - If 6, assumes that 11, 22, 33, 13, 12, and 23 are provided in that order.
    learning_rate : float
        Initial earning rate for optimizer.
    tol : float
        Tolerance for solution error (training terminates if this is reached before the maximum number of
        epochs allowed)
    nepochs : int
        Number of epochs to train for
    init_with_noise : bool
        Whether or not to initialize learnable parameters with random noise; by default, neural network
        parameters are initialized with the Kaiming initialization while physical parameters are initialized
        with 0.
    noise_magnitude : float
        Magnitude of aforementioned noise contribution
    data_type : DataType
        Type of spectra data used. These can be generated from the Kaimal spectra, provided raw as CUSTOM data,
        interpolated, filtered with AUTO, or use the Von Karman model.
    eddy_lifetime : EddyLifetimeType
        Type of model to use for eddy lifetime function. This determines whether a neural network is to be used
        to learn to approximate the function, or if a known model, such as the Mann eddy lifetime is to be used.
    wolfe_iter_count : int
        Sets the number of Wolfe iterations that each step of LBFGS uses
    learn_nu : bool
        If true, learns also the exponent :math:`\nu`, by default True
    """

    num_components: int = 4

    learning_rate: float = 1e-1
    tol: float = 1e-3
    nepochs: int = 10

    init_with_noise: bool = False
    noise_magnitude: float = 1e-3

    eddy_lifetime: EddyLifetimeType = EddyLifetimeType.TAUNET

    wolfe_iter_count: int = 20

    learn_nu: bool = False

    use_learnable_spectrum: bool = False
    p_exponent: float = 4.0  # Defaults to Von Karman values
    q_exponent: float = 17.0 / 6.0  # Defaults to Von Karman values


@dataclass
class PhysicalParameters:
    r"""Define physical parameters for the learning problem.

    This class provides a convenient method of storing and passing around
    the physical parameters required to define a problem; this also offers
    generic default values.

    Args
    ----
    L : float
        Characteristic length scale of the problem; 0.59 for Kaimal
    Gamma : float
        Characteristic time scale of the problem; 3.9 for Kaimal
    sigma : float
        Spectrum amplitude; 3.2 for Kaimal
    Uref : float, optional
        Reference velocity value at hub height (m/s)
    zref : float, optional
        Reference height value; should be measured at hub height (meters)
    domain : torch.Tensor
        :math:`k_1` domain over which spectra data are defined.
    alpha_low : float, optional
        Low wavenumber asymptotic slope for energy spectrum, by default 4.0 (von Karman)
    alpha_high : float, optional
        High wavenumber asymptotic slope for energy spectrum, by default -5.0/3.0 (von Karman)
    transition_slope : float, optional
        Transition slope parameter for energy spectrum, by default 17.0/3.0 (von Karman)
    use_parametrizable_spectrum : bool, optional
        Whether to use the parametrizable energy spectrum, by default False
    """

    L: float
    Gamma: float
    sigma: float

    Uref: float = 10.0
    zref: float = 1.0

    ustar: float = 1.0

    k_inf_asymptote: float = -2.0 / 3.0

    # Energy spectrum asymptotic slope parameters
    alpha_low: float = 4.0  # Low k asymptote (von Karman default)
    alpha_high: float = -5.0 / 3.0  # High k asymptote (von Karman default)
    transition_slope: float = 17.0 / 3.0  # Transition parameter (von Karman default)
    use_parametrizable_spectrum: bool = False  # Whether to use parametrizable spectrum

    domain: torch.Tensor = torch.logspace(-1, 2, 20)


@dataclass
class LossParameters:
    r"""Set coefficients for loss function terms.

    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values, which result in the loss function
    consisting purely of an MSE loss.

    .. note::
        Using the regularization term :math:`\beta` requires a neural network-based approximation to the eddy
        lifetime function.

    Args
    ----
    alpha_pen1 : float
        Set the first-order penalization term coefficient :math:`\alpha_1`, by default 0.0
    alpha_pen2 : float
        Set the second-order penalization term coefficient :math:`\alpha_2`, by default 0.0
    beta_reg : float
        Set the regularization term coefficient :math:`\beta`, by default 0.0
    """

    alpha_pen1: float = 0.0
    alpha_pen2: float = 0.0

    beta_reg: float = 0.0

    gamma_coherence: float = 0.0


@dataclass
class NNParameters:
    r"""Define neural network architecture.

    This class provides a generic and convenient method of storing and passing
    around values required for the definition of the different neural networks
    that are implemented in this package; this also offers default values.

    Args
    ----
    nlayers : int
        Number of layers to be used in neural network model
    input_size : int
        Size of input spectra vector (typically just 3).
    hidden_layer_size : int
        Determines widths of network layers if they are constant.
    hidden_layer_sizes : List[int]
        Determines widths of network layers (input-output pairs must match); used for CustomNet
    activations : List[torch.Module]
        List of activation functions. The list should have the same length as the number of layers, otherwise
        the activation functions begin to repeat from the beginning of the list.
    output_size: int
        Dimensionality of the output vector (typically 3 for spectra-fitting tasks).
    """

    nlayers: int = 2
    input_size: int = 3

    hidden_layer_size: int = 10
    hidden_layer_sizes: list[int] = field(default_factory=list)
    activations: list[nn.Module] = field(default_factory=list)

    output_size: int = 3


#######################################################################################################
# 	Fluctuation field generation parameters
#######################################################################################################


@dataclass
class DomainParameters:
    r"""Define domain parameters for the fluctuation field generation component.

    This class provides a convenient method of storing and passing around
    the domain parameters required for the fluctuation field generation.

    Args
    ----
    grid_dimensions : Tuple[float, float, float]
        The dimensions of the grid in the x, y, and z directions.
    grid_levels : Tuple[int, int, int]
        The number of grid points in the x, y, and z directions.
        These are calculated as :math:`2^{\text{grid\_levels}} + 1`.
    """

    d_dimensions: Union[tuple[float, float, float], "np.ndarray", list[float]]
    d_levels: Union[tuple[int, int, int], "np.ndarray", list[int]]

    def __post_init__(self):
        # Check grid dimensions
        if isinstance(self.d_dimensions, (list, np.ndarray)) and len(self.d_dimensions) != 3:
            raise ValueError("Grid dimensions must be length 3.")

        if isinstance(self.d_dimensions, np.ndarray):
            self._grid_dimensions = self.d_dimensions
        else:
            self._grid_dimensions = np.array(self.d_dimensions)

        # Check grid levels
        if isinstance(self.grid_levels, (list, np.ndarray)) and len(self.grid_levels) != 3:
            raise ValueError("Grid levels must be a tuple of length 3.")

        if isinstance(self.d_levels, np.ndarray):
            self._grid_levels = self.d_levels
        else:
            self._grid_levels = np.array(self.d_levels)

        del self.d_dimensions
        del self.d_levels

    @property
    def grid_dimensions(self):
        """Getter for grid dimensions."""
        return self._grid_dimensions

    @property
    def grid_levels(self):
        """Getter for grid levels."""
        return self._grid_levels


@dataclass
class LowFreqParameters:
    r"""Define parameters for the low-frequency model extension.

    This class provides a convenient method of storing and passing around
    the physical parameters required for the low-frequency model extension.

    The default values are taken from the Syed-Mann (2024) paper.
    """

    L_2D: float = 15_000.0
    sigma2: float = 0.6
    z_i: float = 500.0
    psi_degs: float = 43.0

    c: float | None = None
