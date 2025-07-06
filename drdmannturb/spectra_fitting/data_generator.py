"""Generate one-point spectra data.

This module contains the ``OnePointSpectraDataGenerator`` class, which generates one-point spectra data for a given
set of parameters.

.. note:: This module does NOT contain any examples which provide generated spectral coherence data.
"""

from pathlib import Path

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

    ops_values = ops_values * k1.unsqueeze(-1).unsqueeze(-1)

    # TODO: Implement spectral coherence generation

    return {"k1": k1, "ops": ops_values, "coherence": None}


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

    ops_values = torch.zeros([len(k1), 3, 3], dtype=k1.dtype)

    ops_values[:, 0, 0] = 52.5 * n / (1 + 33 * n) ** (5 / 3)
    ops_values[:, 1, 1] = 8.5 * n / (1 + 9.5 * n) ** (5 / 3)
    ops_values[:, 2, 2] = 1.05 * n / (1 + 5.3 * n ** (5 / 3))
    ops_values[:, 0, 2] = -7 * n / (1 + 9.6 * n) ** (12.0 / 5.0)

    # TODO: Implement spectral coherence generation

    return {
        "k1": k1,
        "ops": ops_values * torch.tensor(ustar**2, dtype=k1.dtype),
        "coherence": None,
    }


# TODO: Custom data generator should be deprecated in favor of requiring the user to
#       the correct data format(s) themselves.
#
#       We will split this file into data/generators.py and data/cleaners.py,
#       where generators is self-explanatory and cleaners will provide a variety of routines
#       for cleaning data and performing "generic computations" on the data.
#
#       This API/interface was just impossible to understand and use.
class CustomDataFormatter:
    """Custom data formatter.

    Given a one point spectra data file (CSV) formatted as:

    .. code-block:: text

        f, F11(f), F22(f), F33(f), F13(f), F23(f), F12(f)

    TODO: Add in functionalities for loading spectral coherence data.
    TODO: This should also accept NetCDF and other formats.
    """

    data_file: Path | str
    k1_domain: torch.Tensor
    CustomData: torch.Tensor

    def __init__(
        self,
        data_file: Path | str,
    ):
        _data_file = Path(data_file)
        if not _data_file.exists():
            raise FileNotFoundError(f'Provided data file path "{_data_file}" does not exist.')

        # Check that the file is a CSV
        if _data_file.suffix not in [".csv", ".dat"]:
            raise ValueError(f'Provided data file path "{_data_file}" is not a CSV file.')

    def _load_data(self) -> np.ndarray:
        pass

    def format_data(self) -> dict[str, torch.Tensor]:
        r"""Provide a correctly formatted data dictionary based on provided data file."""
        DataValues = torch.zeros([len(self.k1_domain), 3, 3])

        DataValues[:, 0, 0] = self.CustomData[:, 1]  # uu
        DataValues[:, 1, 1] = self.CustomData[:, 2]  # vv
        DataValues[:, 2, 2] = self.CustomData[:, 3]  # ww
        # NOTE: is always negative
        DataValues[:, 0, 2] = -self.CustomData[:, 4]  # uw
        # TODO: Can be negative, skipping for now
        DataValues[:, 1, 2] = self.CustomData[:, 5]  # vw
        # TODO: Can be negative, skipping for now
        DataValues[:, 0, 1] = self.CustomData[:, 6]  # uv

        return {
            "k1": self.k1_domain,
            "ops": DataValues,
            "coherence": None,
        }
