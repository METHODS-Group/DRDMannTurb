"""
This module defines several dataclasses that comprise the set-up for a calibration problem of a DRD-Mann model. 
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from drdmannturb.shared.enums import DataType, EddyLifetimeType, PowerSpectraType


@dataclass
class ProblemParameters:
    """
    This class provides a convenient method of storing and passing around
    generic numerical parameters; this also offers default values

    Fields
    ------
    learning_rate : float
        Learning rate for optimizer.
    tol : float
        Tolerance for solution error (training terminates once this is reached)
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
        TODO: check this
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
    """
    This class provides a convenient method of storing and passing around
    the physical parameters required to define a problem; this also offers
    generic default values.

    Fields
    ------
    L : float
        _description_
    Gamma
        _description_
    sigma
        _description_
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
    """
    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values.

    Fields
    ------
    alpha_pen : float
        Set the alpha penalization term coefficient, by default 0.0
    alpha_reg : float
        Set the alpha regularization term coefficient, by default 0.0
    beta_pen : float
        Set the beta penalization term coefficient, by default 0.0
    """

    alpha_pen: float = 0.0
    alpha_reg: float = 0.0

    beta_pen: float = 0.0


@dataclass
class NNParameters:
    """
    This class provides a generic and convenient method of storing and passing
    around values required for the definition of the different neural networks
    that are implemented in this package; this also offers default values.

    Fields
    ------
    nlayers : int
        Number of layers to be used in neural network model
    input_size : int
        Size of input spectra vector (typically just 3)
    hidden_layer_size : int
        Determines widths of network layers if they are constant
    hidden_layer_sizes : List[int]
        Determines widths of network layers (input-output pairs must match); used for CustomNet or ResNet models
    activations : List[str]
        List of activation functions, matching TODO: why not make this a list of nn.modules?
    output_size: int

    """

    nlayers: int = 2
    input_size: int = 3

    # TODO -- better way of doing this?
    hidden_layer_size: int = 3
    hidden_layer_sizes: List[int] = field(
        default_factory=list
    )  # should be used for customnet or resnet
    # [10, 10]
    activations: List[str] = field(default_factory=list)
    # ["relu", "relu"]
    n_modes: int = 10

    output_size: int = 3
