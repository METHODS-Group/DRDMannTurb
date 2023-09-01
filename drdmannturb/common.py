from typing import Union

import numpy as np
import torch
from scipy.special import hyp2f1


@torch.jit.script
def VKEnergySpectrum(kL: torch.Tensor) -> torch.Tensor:
    """
    Von Karman energy spectrum (without scaling)
    """

    return kL**4 / (1.0 + kL**2) ** (17.0 / 6.0)


def MannEddyLifetime(kL: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Numpy implementation of Mann's eddy lifetime
    """
    # print("!" * 30 + "Called CPU again :(")
    x = kL.cpu().detach().numpy() if torch.is_tensor(kL) else kL
    y = x ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(x ** (-2))))
    y = torch.tensor(y, dtype=torch.float64) if torch.is_tensor(kL) else y
    return y
