"""
This module implements the RDT Power Spectra.  
"""

import torch

Tensor = torch.Tensor


@torch.jit.script
def PowerSpectraRDT(
    k: Tensor, beta: Tensor, E0: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Classical rapid distortion spectra, which is the solution to

    .. math::
        \frac{\bar{D} \mathrm{~d} Z_j(\boldsymbol{k}, t)}{\bar{D} t}=\frac{\partial U_{\ell}}{\partial x_k}\left(2 \frac{k_j k_{\ell}}{k^2}-\delta_{j \ell}\right) \mathrm{d} Z_k(\boldsymbol{k}, t)

    given by

    .. math::
        \mathrm{d} \mathbf{Z}(\boldsymbol{k}(t), t)=\boldsymbol{D}_\tau(\boldsymbol{k}) \mathrm{d} \mathbf{Z}\left(\boldsymbol{k}_0, 0\right).

    Refer to the original DRD paper, Section III, subsection B for a full expansion.

    Parameters
    ----------
    k : torch.Tensor
        Wave vector domain.
    beta : torch.Tensor
        Evaluated eddy lifetime function.
    E0 : torch.Tensor
        Evaluated and non-dimensionalized von Karman energy spectrum.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        6-tuple of the components of the velocity-spectrum tensor in the order: :math:`\Phi_{11}, \Phi_{22}, \Phi_{33}, \Phi_{13}, \Phi_{12}, \Phi_{23}`.
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

    zeta1 = C1 - k2 / k1 * C2
    zeta2 = C1 * k2 / k1 + C2
    E0 /= 4 * torch.pi
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
