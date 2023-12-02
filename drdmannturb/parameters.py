"""
This module defines several dataclasses that comprise the set-up for a calibration problem of a DRD-Mann model. 
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn

from .enums import DataType, EddyLifetimeType, PowerSpectraType

__all__ = ["ProblemParameters", "PhysicalParameters", "NNParameters", "LossParameters"]


@dataclass
class ProblemParameters:
    r"""
    This class provides a convenient method of storing and passing around
    generic numerical parameters; this also offers default values

    Fields
    ------
    learning_rate : float
        Initial earning rate for optimizer.
    tol : float
        Tolerance for solution error (training terminates if this is reached before the maximum number of epochs allowed)
    nepochs : int
        Number of epochs to train for
    init_with_noise : bool
        Whether or not to initialize with random noise TODO: check this
    noise_magnitude : float
        Magnitude of aforementioned noise contribution
    fg_coherence : bool
        description here TODO: check this
    data_type : DataType
        Type of TODO: check this
    eddy_lifetime : EddyLifetimeType
        Type of model to use for eddy lifetime function
    power_spectra : PowerSpectraType
        Type of model to use for power spectra
    wolfe_iter_count : int
        Sets the number of Wolfe iterations that each step of LBFGS uses
    learn_nu : bool
        If true, learns also the exponent :math:`\nu`, by default True
    """

    learning_rate: float = 1e-1
    tol: float = 1e-3
    nepochs: int = 10

    init_with_noise: bool = False
    noise_magnitude: float = 1e-3

    fg_coherence: bool = False
    data_type: DataType = DataType.KAIMAL
    eddy_lifetime: EddyLifetimeType = EddyLifetimeType.CUSTOMMLP
    power_spectra: PowerSpectraType = PowerSpectraType.RDT

    wolfe_iter_count: int = 20

    learn_nu: bool = False


@dataclass
class PhysicalParameters:
    r"""
    This class provides a convenient method of storing and passing around
    the physical parameters required to define a problem; this also offers
    generic default values.

    Fields
    ------
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
    Iref : float, optional
        Longitudinal turbulence scale parameter at hub height (meters)
    """

    L: float
    Gamma: float
    sigma: float

    Uref: float = 10.0
    zref: float = 1.0
    Iref: float = 0.14

    sigma1: float = Iref * (0.75 * Uref + 5.6)
    Lambda1: float = 42.0

    z0: float = 0.01
    ustar: float = 0.41 * Uref / np.log(zref / z0)

    domain: torch.Tensor = torch.logspace(-1, 2, 20)


@dataclass
class LossParameters:
    r"""
    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values, which result in the loss function consisting purely of an MSE loss.

    .. note::
        Using the regularization term :math:`\beta` requires a neural network-based approximation to the eddy lifetime function.

    Fields
    ------
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


@dataclass
class NNParameters:
    r"""
    This class provides a generic and convenient method of storing and passing
    around values required for the definition of the different neural networks
    that are implemented in this package; this also offers default values.

    Fields
    ------
    nlayers : int
        Number of layers to be used in neural network model
    input_size : int
        Size of input spectra vector (typically just 3).
    hidden_layer_size : int
        Determines widths of network layers if they are constant.
    hidden_layer_sizes : List[int]
        Determines widths of network layers (input-output pairs must match); used for CustomNet
    activations : List[torch.Module]
        List of activation functions. The list should have the same length as the number of layers, otherwise the activation functions begin to repeat from the beginning of the list.
    output_size: int
        Dimensionality of the output vector (typically 3 for spectra-fitting tasks).
    """

    nlayers: int = 2
    input_size: int = 3

    hidden_layer_size: int = 10
    hidden_layer_sizes: List[int] = field(default_factory=list)
    activations: List[nn.Module] = field(default_factory=list)  # [nn.ReLU(), nn.ReLU()]

    output_size: int = 3
