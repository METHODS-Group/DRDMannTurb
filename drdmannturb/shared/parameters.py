"""
This module defines several dataclasses for ease of parameter definition
and interpass.
"""

from dataclasses import dataclass, field
from typing import List

from drdmannturb.shared.enums import DataType, EddyLifetimeType, PowerSpectraType

import numpy as np
import torch


@dataclass
class ProblemParameters:
    """
    This class provides a convenient method of storing and passing around
    generic numerical parameters; this also offers default values
    """

    learning_rate: float = 1e-1
    tol: float = 1e-3
    nepochs: int = 100

    init_with_noise: bool = False
    noise_magnitude: float = 1e-3

    fg_coherence: bool = False
    data_type: DataType = DataType.KAIMAL
    eddy_lifetime: EddyLifetimeType = EddyLifetimeType.CUSTOMMLP
    power_spectra: PowerSpectraType = PowerSpectraType.RDT

    learn_nu: bool = False  # true in exp 2


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

    domain: torch.Tensor = torch.logspace(
        -1, 2, 20
    ) # exp 2 should be logspcae (-2, 2, 40)


@dataclass
class LossParameters:
    """
    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values.

    Parameters
    ----------
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
    that are implemented in this package; this also offers default values
    """

    nlayers: int = 2
    input_size: int = 3

    # TODO -- better way of doing this?
    hidden_layer_size: int = 3
    hidden_layer_sizes: List[int] = field(default_factory=list) # should be used for customnet or resnet
    # [10, 10]
    activations: List[str] = field(default_factory=list)
    # ["relu", "relu"]
    output_size: int = 3
