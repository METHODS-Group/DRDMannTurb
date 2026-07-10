"""Several dataclasses that make it easy to pass around parameters."""

from dataclasses import dataclass

__all__ = ["IntegrationParameters", "LossParameters"]


@dataclass
class LossParameters:
    r"""Set coefficients for loss function terms.

    This class provides a convenient method of storing and passing around
    the loss function term coefficients; this also offers default values, which result in the loss function
    consisting purely of an MSE loss.

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
class IntegrationParameters:
    """Parameters for integration grids in spectra calculations.

    This dataclass defines the parameters for creating log-spaced grids
    used in one-point spectra and coherence integrations.

    Attributes
    ----------
        ops_log_min: Minimum exponent for OPS grid (default -3).
        ops_log_max: Maximum exponent for OPS grid (default 3).
        ops_num_points: Number of points per side for OPS grid (default 100).
        coh_log_min: Minimum exponent for coherence grid (default -3).
        coh_log_max: Maximum exponent for coherence grid (default 3).
        coh_num_points: Number of points per side for coherence grid (default 100).
    """

    ops_log_min: float = -3.0
    ops_log_max: float = 3.0
    ops_num_points: int = 100
    coh_log_min: float = -3.0
    coh_log_max: float = 3.0
    coh_num_points: int = 100
