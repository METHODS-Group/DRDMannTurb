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
    """
    Torch and Numpy implementation of Mann's Eddy Lifetime

    Parameters
    ----------
    kL : Union[torch.Tensor, np.ndarray]
        Scaled wave number

    Returns
    -------
    torch.Tensor
        Result of the evaluation
    """
    x = kL.cpu().detach().numpy() if torch.is_tensor(kL) else kL
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(x ** (-2))))
    y = torch.tensor(y, dtype=torch.float64) if torch.is_tensor(kL) else y

    return y
