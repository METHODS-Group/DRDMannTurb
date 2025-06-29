"""Covariance kernel implementation for neural network models."""

import numpy as np
import torch

from ..spectra_fitting import OnePointSpectra
from .covariance_kernels import Covariance


class NNCovariance(Covariance):
    r"""
    Neural Network covariance kernel.

    Like other covariance kernel implementations, this evaluates the
    :math:`G(\boldsymbol{k})` which satisfies :math:`G(\boldsymbol{k}) G^*(\boldsymbol{k})=\Phi(\boldsymbol{k},
    \tau(\boldsymbol{k}))` where the spectral tensor is defined through the eddy lifetime function learned by the neural
    network as well as the fitted spectra. Here,

    .. math::
            \tau(\boldsymbol{k})=\frac{T|\boldsymbol{a}|^{\nu-\frac{2}{3}}}
            {\left(1+|\boldsymbol{a}|^2\right)^{\nu / 2}}, \quad \boldsymbol{a}=\boldsymbol{a}(\boldsymbol{k})

    satisfies

    .. math::
        :nowrap:

        \begin{align}
            \Phi(\boldsymbol{k}, \tau) & =\left\langle\widehat{\mathbf{u}}(\boldsymbol{k})
                \widehat{\mathbf{u}}^*(\boldsymbol{k})\right\rangle \\
            & =\mathbf{D}_\tau(\boldsymbol{k}) \boldsymbol{G}_0\left(\boldsymbol{k}_0\right)\left\langle
                \widehat{\boldsymbol{\xi}}\left(\boldsymbol{k}_0\right) \widehat{\boldsymbol{\xi}}^*
                \left(\boldsymbol{k}_0\right)\right\rangle \boldsymbol{G}_0^*\left(\boldsymbol{k}_0\right)
                \mathbf{D}_\tau^*(\boldsymbol{k}) \\
            & =\mathbf{D}_\tau(\boldsymbol{k}) \boldsymbol{G}_0\left(\boldsymbol{k}_0\right) \boldsymbol{G}_0^*
                \left(\boldsymbol{k}_0\right) \mathbf{D}_\tau^*(\boldsymbol{k}) \\
            & =\mathbf{D}_\tau(\boldsymbol{k}) \Phi^{\mathrm{VK}}\left(\boldsymbol{k}_0\right)
                \mathbf{D}_\tau^*(\boldsymbol{k}) .
        \end{align}

    For more detailed definitions of individual terms, refer to section III B (specifically pages 4 and 5) of
    the original DRD paper.
    """

    def __init__(
        self,
        ndim: int,
        length_scale: float,
        E0: float,
        Gamma: float,
        ops: OnePointSpectra,
        h_ref: float,
    ):
        """Initialize the neural network covariance kernel.

        Parameters
        ----------
        ndim : int, optional
            Number of dimensions for kernel to operate over.
        length_scale : float, optional
            Length scale non-dimensionalizing constant.
        E0 : float, optional
            Energy spectrum.
        Gamma : float, optional
            Time scale.
        ops : OnePointSpectra
            Pre-trained OnePointSpectra object with a neural network representing the eddy lifetime function and
            containing the non-dimensionalizing scales, which are also learned by the DRD model.
        h_ref : float
            Reference height (this is not a parameter learned by the DRD model).

        Raises
        ------
        ValueError
            ndim must be 3 for NN covariance.
        """
        super().__init__()

        ### Spatial dimensions
        if ndim != 3:
            raise ValueError("ndim must be 3 for NN covariance.")

        self.ndim = 3

        self.L = length_scale
        self.E0 = E0
        self.Gamma = Gamma

        self.OPS = ops
        self.h_ref = h_ref
        ### NOTE: NN implicitly involves the length scales (it is associated with non-dimensional internal L)
        ### NOTE: However, here we scale L with the reference_height - the latter has to be taken into account from
        ###       the physical setting

    def precompute_Spectrum(self, Frequencies: np.ndarray) -> np.ndarray:
        """Pre-compute the square-root of the associated spectrum tensor in the complex domain.

        Parameters
        ----------
        Frequencies : np.ndarray
            Frequency domain in 3D over which to compute the square-root of the spectral tensor.

        Returns
        -------
        np.ndarray
            Square-root of the spectral tensor evaluated in the frequency domain; note that these are complex values.
        """
        Nd = [Frequencies[j].size for j in range(self.ndim)]
        SqrtSpectralTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))
        tmpTens = np.tile(np.zeros(Nd), (3, 3, 1, 1, 1))

        k = np.array(list(np.meshgrid(*Frequencies, indexing="ij")))
        kk = np.sum(k**2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            k_torch = torch.tensor(np.moveaxis(k, 0, -1)) * self.h_ref
            beta_torch = self.Gamma * self.OPS.tauNet(k_torch)
            beta = beta_torch.detach().cpu().numpy() if beta_torch.is_cuda else beta_torch.detach().numpy()
            beta[np.where(kk == 0)] = 0

            k1 = k[0, ...]
            k2 = k[1, ...]
            k3 = k[2, ...]
            k30 = k3 + beta * k1

            kk0 = k1**2 + k2**2 + k30**2

            #### Isotropic with k0

            const = self.E0 * (self.L ** (17 / 3)) / (4 * np.pi)
            const = np.sqrt(const / (1 + (self.L**2) * kk0) ** (17 / 6))

            # to enforce zero mean in the x-direction:
            # const[k1 == 0] = 0.0

            tmpTens[0, 1, ...] = -const * k30
            tmpTens[0, 2, ...] = const * k2
            tmpTens[1, 0, ...] = const * k30
            tmpTens[1, 2, ...] = -const * k1
            tmpTens[2, 0, ...] = -const * k2
            tmpTens[2, 1, ...] = const * k1

            #### RDT

            s = k1**2 + k2**2
            C1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (kk * s)
            numerator = beta * k1 * np.sqrt(s)
            denominator = kk0 - k30 * k1 * beta
            C2 = k2 * kk0 / s ** (3 / 2) * np.arctan2(numerator, denominator)

            zeta1 = C1 - k2 / k1 * C2
            zeta2 = k2 / k1 * C1 + C2
            zeta3 = kk0 / kk

            # deal with divisions by zero
            zeta1 = np.nan_to_num(zeta1)
            zeta2 = np.nan_to_num(zeta2)
            zeta3 = np.nan_to_num(zeta3)

            SqrtSpectralTens[0, 0, ...] = tmpTens[0, 0, ...] + zeta1 * tmpTens[2, 0, ...]
            SqrtSpectralTens[0, 1, ...] = tmpTens[0, 1, ...] + zeta1 * tmpTens[2, 1, ...]
            SqrtSpectralTens[0, 2, ...] = tmpTens[0, 2, ...] + zeta1 * tmpTens[2, 2, ...]
            SqrtSpectralTens[1, 0, ...] = tmpTens[1, 0, ...] + zeta2 * tmpTens[2, 0, ...]
            SqrtSpectralTens[1, 1, ...] = tmpTens[1, 1, ...] + zeta2 * tmpTens[2, 1, ...]
            SqrtSpectralTens[1, 2, ...] = tmpTens[1, 2, ...] + zeta2 * tmpTens[2, 2, ...]
            SqrtSpectralTens[2, 0, ...] = zeta3 * tmpTens[2, 0, ...]
            SqrtSpectralTens[2, 1, ...] = zeta3 * tmpTens[2, 1, ...]
            SqrtSpectralTens[2, 2, ...] = zeta3 * tmpTens[2, 2, ...]

            return SqrtSpectralTens * 1j
