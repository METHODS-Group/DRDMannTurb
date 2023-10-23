"""
This module implements the RDT PowerSpectra type
"""

import numpy as np
import torch


@torch.jit.script
def PowerSpectraRDT(
    k: torch.Tensor, beta: torch.Tensor, E0
) -> tuple[torch.Tensor, ...]:
    """
    Classical rapid distortion spectra

    Parameters
    ----------
    k : torch.Tensor
        Wave vector input
    beta : torch.Tensor
        _description_
    E0 : _type_
        _description_

    Returns
    -------
    tuple[torch.Tensor, ...]
        6-tuple of the components of the velocity-spectrum tensor
    """

    k1, k2, k3 = k[..., 0], k[..., 1], k[..., 2]

    k30 = k3 + beta * k1
    kk0 = k1**2 + k2**2 + k30**2
    kk = k1**2 + k2**2 + k3**2
    s = k1**2 + k2**2

    C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
    C2 = (
        k2
        * kk0
        / torch.sqrt(s**3)
        * torch.atan2(beta * k1 * torch.sqrt(s), kk0 - k30 * k1 * beta)
    )

    # arg1 = k30/torch.sqrt(s)
    # arg2 = k3 /torch.sqrt(s)
    # C2  = k2 * kk0 / torch.sqrt(s**3) * (torch.atan(arg1) - torch.atan(arg2))

    zeta1 = C1 - k2 / k1 * C2
    zeta2 = C1 * k2 / k1 + C2
    E0 /= 4 * np.pi
    Phi11 = (
        E0
        / (kk0**2)
        * (kk0 - k1**2 - 2 * k1 * k30 * zeta1 + (k1**2 + k2**2) * zeta1**2)
    )
    Phi22 = (
        E0
        / (kk0**2)
        * (kk0 - k2**2 - 2 * k2 * k30 * zeta2 + (k1**2 + k2**2) * zeta2**2)
    )
    Phi33 = E0 / (kk**2) * (k1**2 + k2**2)
    Phi13 = E0 / (kk * kk0) * (-k1 * k30 + (k1**2 + k2**2) * zeta1)

    Phi12 = (
        E0
        / (kk0**2)
        * (
            -k1 * k2
            - k1 * k30 * zeta2
            - k2 * k30 * zeta1
            + (k1**2 + k2**2) * zeta1 * zeta2
        )
    )
    Phi23 = E0 / (kk * kk0) * (-k2 * k30 + (k1**2 + k2**2) * zeta2)

    return Phi11, Phi22, Phi33, Phi13, Phi12, Phi23
