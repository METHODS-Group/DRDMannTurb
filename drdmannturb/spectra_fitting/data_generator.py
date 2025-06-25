"""Generate one-point spectra data.

This module contains the ``OnePointSpectraDataGenerator`` class, which generates one-point spectra data for a given
set of parameters.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def generate_von_karman_spectra(k1: torch.Tensor, L: float = 0.59, C: float = 3.2) -> torch.Tensor:
    """Generate von Karman spectra data.

    .. note:: This is already frequency-weighted. DRD models assume that the provided data
        is frequency-weighted.

    Parameters
    ----------
    k1 : torch.Tensor
        Wavevector domain
    L : float, optional
        Length scale, by default 0.59
    C : float, optional
        Constant, by default 3.2
    """
    # Vectorized computation
    k1_squared = k1**2
    L_inv_squared = L ** (-2)
    denominator = L_inv_squared + k1_squared

    # Initialize tensor with zeros
    ops_values = torch.zeros([len(k1), 3, 3])

    # Compute diagonal elements vectorized
    ops_values[:, 0, 0] = 9 / 55 * C / denominator ** (5 / 6)
    ops_values[:, 1, 1] = 3 / 110 * C * (3 * L_inv_squared + 8 * k1_squared) / denominator ** (11 / 6)
    ops_values[:, 2, 2] = 3 / 110 * C * (3 * L_inv_squared + 8 * k1_squared) / denominator ** (11 / 6)

    # Multiply by k1 (broadcasting across the 3x3 matrices)
    ops_values = ops_values * k1.unsqueeze(-1).unsqueeze(-1)

    # TODO: Implement spectral coherence generation

    return ops_values


def generate_kaimal_spectra(k1: torch.Tensor, zref: float, ustar: float) -> torch.Tensor:
    """Generate Kaimal spectra data.

    .. note:: This is already frequency-weighted. DRD models assume that the provided data
        is frequency-weighted.

    Parameters
    ----------
    k1 : torch.Tensor
        Wavevector domain
    zref : float
        Reference altitude
    ustar : float
        Friction velocity

    Returns
    -------
    torch.Tensor
        Spectral tensor data
    """
    n = 1 / (2 * np.pi) * k1 * zref

    ops_values = torch.zeros([len(k1), 3, 3])

    ops_values[:, 0, 0] = 52.5 * n / (1 + 33 * n) ** (5 / 3)
    ops_values[:, 1, 1] = 8.5 * n / (1 + 9.5 * n) ** (5 / 3)
    ops_values[:, 2, 2] = 1.05 * n / (1 + 5.3 * n ** (5 / 3))
    ops_values[:, 0, 2] = -7 * n / (1 + 9.6 * n) ** (12.0 / 5.0)

    # TODO: Implement spectral coherence generation

    return {
        "ops": ops_values * ustar**2,
        "coherence": torch.zeros([len(k1), 3, 3]),
    }


# TODO: Custom data generator should be deprecated in favor of requiring the user to
#       the correct data format(s) themselves.
#
#       We will split this file into data/generators.py and data/cleaners.py,
#       where generators is self-explanatory and cleaners will provide a variety of routines
#       for cleaning data and performing "generic computations" on the data.
#
#       This API/interface was just impossible to understand and use.
class CustomDataGenerator:
    r"""One point spectra data generator.

    The one point spectra data generator, which evaluates one of a few spectral tensor models across a grid of
      :math:`k_1` wavevector points which is used as data in the fitting done in the :py:class:`OnePointSpectra` class.

    The type of spectral tensor is determined by the :py:enum:`DataType` argument, which determines one of the
    following models:

    #. ``DataType.CUSTOM``, usually used for data that is processed from real-world data. The spectra values are to be
        provided as the ``spectra_values`` field, or else to be loaded from a provided ``spectra_file``. The result is
        that the provided data are matched on the wavevector domain.
    """

    def __init__(
        self,
        zref: float,
        ustar: float = 1.0,
        data_points: Optional[Sequence[tuple[torch.tensor, float]]] = None,
        k1_data_points: Optional[torch.Tensor] = None,
        spectra_values: Optional[torch.Tensor] = None,
        spectra_file: Optional[Union[Path, str]] = None,
        seed: int = 3,
    ):
        r"""Initialize the OnePointSpectraDataGenerator.

        Parameters
        ----------
        zref : float
            Reference altitude value
        ustar : float
            Friction velocity, by default 1.0.
        data_points : Iterable[Tuple[torch.tensor, float]], optional
            Observed spectra data points at each of the :math:`k_1` coordinates, paired with the associated reference
            height (typically kept at 1, but may depend on applications).
        data_type : DataType, optional
            Indicates the data format to generate and operate with, by
            default ``DataType.KAIMAL``
        k1_data_points : Optional[Any], optional
            Wavevector domain of :math:`k_1`, by default None. This is only to be used when the ``AUTO`` tag is chosen
            to define the domain over which a non-linear regression is to be computed. See the interpolation module
            for examples of unifying different :math:`k_1` domains.
        spectra_file : Optional[Path], optional
            If using ``DataType.CUSTOM`` or ``DataType.AUTO``, this
            is used to indicate the data file (a .dat) to read
            from. Since it is not used by others, it is by
            default None

        Raises
        ------
        ValueError
            In the case that ``DataType.CUSTOM`` is indicated, but no spectra_file
            is provided
        ValueError
            Did not provide DataPoints during initialization for DataType method requiring spectra data.
        """
        self.DataPoints = data_points
        self.k1 = k1_data_points

        self.seed = seed

        if spectra_values is not None:
            self.spectra_values = spectra_values

        self.zref = zref
        self.ustar = ustar

        if spectra_file is not None:
            self.CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))
        else:
            raise ValueError("Did not provide a spectra_file argument.")

        return

    def generate_Data(
        self, DataPoints: Sequence[tuple[torch.tensor, float]]
    ) -> tuple[list[tuple[torch.Tensor, float]], torch.Tensor]:
        r"""Generate data from provided configuration.

        Generates a single spectral tensor from provided data or from a surrogate model.
        The resulting tensor is of shape (number of :math:`k_1` points):math:`\times 3 \times 3`,
        ie the result consists of the spectral tensor evaluated across the provided range of
        :math:`k_1` points. The spectra model is set during object instantiation.

        .. note::
            The ``DataType.CUSTOM`` type results in replication of the provided spectra data.

        Parameters
        ----------
        DataPoints : Iterable[Tuple[torch.tensor, float]]
            Observed spectra data points at each of the :math:`k_1` coordinates, paired with the associated reference
            height (typically kept at 1, but may depend on applications).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Evaluated spectral tensor on each of the provided grid points depending on the ``DataType`` selected.

        Raises
        ------
        ValueError
            DataType set such that an iterable set of spectra values is required. This is for any DataType other than
            ``CUSTOM`` and ``AUTO``.
        """
        DataValues = torch.zeros([len(DataPoints), 3, 3])

        DataValues[:, 0, 0] = self.CustomData[:, 1]  # uu
        DataValues[:, 1, 1] = self.CustomData[:, 2]  # vv
        DataValues[:, 2, 2] = self.CustomData[:, 3]  # ww
        # NOTE: is always negative
        DataValues[:, 0, 2] = -self.CustomData[:, 4]  # uw
        # TODO: Can be negative, skipping for now
        DataValues[:, 1, 2] = self.CustomData[:, 5]  # vw
        # TODO: Can be negative, skipping for now
        DataValues[:, 0, 1] = self.CustomData[:, 6]  # uv

        DataPoints = list(zip(DataPoints, [self.zref] * len(DataPoints)))
        self.Data = (DataPoints, DataValues)
        return self.Data
