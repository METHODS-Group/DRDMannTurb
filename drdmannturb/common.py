"""
This module contains implementations of common functions required a few times throughout

TODO -- possibly move this elsewhere/ somewhere more appropriate than here
"""

__all__ = ["VKEnergySpectrum", "MannEddyLifetime", "Mann_linear_exponential_approx"]

from typing import Union

import numpy as np
import torch
from scipy.special import hyp2f1


@torch.jit.script
def VKEnergySpectrum(kL: torch.Tensor) -> torch.Tensor:
    """
    Von Karman energy spectrum (without scaling)

    Parameters
    ----------
    kL : torch.Tensor
        Scaled wave number

    Returns
    -------
    torch.Tensor
        Result of the evaluation
    """
    return kL**4 / (1.0 + kL**2) ** (17.0 / 6.0)


def MannEddyLifetime(kL: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    r"""
    Implementation of the full Mann eddy lifetime function, of the form

    .. math::
        \tau^{\mathrm{IEC}}(k)=\frac{(k L)^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-(k L)^{-2}\right)}}

    This function can execute with input data that are either in Torch or numpy. However,

    .. warning::
        This function depends on SciPy for evaluating the hypergeometric function, meaning a GPU tensor will be returned to the CPU for a single evaluation and then converted back to a GPU tensor. This incurs a substantial loss of performance.

    Parameters
    ----------
    kL : Union[torch.Tensor, np.ndarray]
        Scaled wave number

    Returns
    -------
    torch.Tensor
        Evaluated Mann eddy lifetime function.
    """

    x = kL.cpu().detach().numpy() if torch.is_tensor(kL) else kL
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(x ** (-2))))
    y = torch.tensor(y, dtype=torch.float64) if torch.is_tensor(kL) else y

    return y


def Mann_linear_exponential_approx(
    kL: torch.Tensor, coefficient: torch.Tensor, intercept: torch.Tensor
) -> torch.Tensor:
    r"""A surrogate for the term involving the hypergeometric function

    .. math::
        \frac{x^{-\frac{2}{3}}}{\sqrt{{ }_2 F_1\left(1 / 3,17 / 6 ; 4 / 3 ;-x^{-2}\right)}}

    via an exponential function, which is a reasonable approximation since the resulting :math:`\tau` is nearly linear on a log-log plot. The resulting approximation is the function

    .. math::
        \exp \left( \alpha kL + \beta \right)

    where :math:`\alpha, \beta` are obtained from a linear regression on the hypergeometric function on the domain of interest. In particular, using this function requires that a linear regression has already been performed on the basis of the above function depending on the hypergeometric function, which is an operation performed once on the CPU. The rest of this subroutine is on the GPU and unlike the full hypergeometric approximation, will not incur any slow-down of the rest of the spectra fitting.

    Parameters
    ----------
    kL : torch.Tensor
        _description_
    coefficient : torch.Tensor
        The linear coefficient :math:`\alpha` obtained from the linear regression.
    intercept : torch.Tensor
        The intercept term :math:`\beta` from the linear regression.

    Returns
    -------
    torch.Tensor
        Exponential approximation to the Mann eddy lifetime function output.
    """

    return torch.exp(coefficient * torch.log(kL) + intercept)
