"""Data generation and formatting for the model calibration module.

.. note:: This does NOT contain any examples which provide generated spectral coherence data.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch


def generate_von_karman_spectra(k1: torch.Tensor, L: float = 0.59, C: float = 3.2) -> dict[str, pl.DataFrame | None]:
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

    freq = k1.tolist()
    ops_data = {
        "freq": freq,
        "uu": ops_values[:, 0, 0].tolist(),
        "vv": ops_values[:, 1, 1].tolist(),
        "ww": ops_values[:, 2, 2].tolist(),
        "uw": [float("nan")] * len(freq),
        "vw": [float("nan")] * len(freq),
        "uv": [float("nan")] * len(freq),
    }
    df = pl.DataFrame(ops_data)
    # TODO: Implement spectral coherence generation

    return {"ops": df, "coherence": None}


def generate_kaimal_spectra(
    k1: torch.Tensor,
    zref: float,
    ustar: float,
    dtype: type[pl.DataType] = pl.Float64,
) -> dict[str, pl.DataFrame | None]:
    """Generate Kaimal spectra data.

    TODO: Write about the data.

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
    freq = (k1 * zref) / (2 * np.pi)

    # TODO: No reason to use the tensor here at all
    ops_values = torch.zeros([len(k1), 3, 3], dtype=k1.dtype)

    ops_values[:, 0, 0] = 52.5 * freq / (1 + 33 * freq) ** (5 / 3)
    ops_values[:, 1, 1] = 8.5 * freq / (1 + 9.5 * freq) ** (5 / 3)
    ops_values[:, 2, 2] = 1.05 * freq / (1 + 5.3 * freq ** (5 / 3))
    ops_values[:, 0, 2] = -7 * freq / (1 + 9.6 * freq) ** (12.0 / 5.0)

    freq = freq.tolist()
    ops_data = {
        "freq": freq,
        "uu": ops_values[:, 0, 0].tolist(),
        "vv": ops_values[:, 1, 1].tolist(),
        "ww": ops_values[:, 2, 2].tolist(),
        "uw": ops_values[:, 0, 2].tolist(),
        "vw": [float("nan")] * len(freq),
        "uv": [float("nan")] * len(freq),
    }
    df = pl.DataFrame(ops_data)

    # TODO: Implement spectral coherence generation

    return {
        "ops": df,
        "coherence": None,
    }


class CustomDataLoader:
    """Custom data loader.

    This class is used to load one-point spectra and coherence data from a CSV file.

    The CSV file should be formatted as:

    .. code-block:: text

        f, F11(f), F22(f), F33(f), F13(f), F23(f), F12(f)

    where 'f' is the frequency, and F_ij(f) is the *frequency-weighted* one-point spectra. This class must be
    able to find these quantities as

    .. code-block:: text

        freq, uu, vv, ww, uw, vw, uv

    Here, uu, vv, ww (the auto-spectra components) are required. uw, vw, uv are optional. Any missing cross-spectra
    components are set to NaN and ignored during training.

    TODO: Write about the coherence data format.
    """

    # dtype to load the data as
    dtype: type[pl.DataType]

    # Data file paths
    ops_data_file: Path
    coherence_data_file: Path | None

    # Data storage
    ops_data_df: pl.DataFrame
    coh_data_df: pl.DataFrame | None

    def __init__(
        self,
        ops_data_file: Path | str,
        coherence_data_file: Path | str | None = None,
        dtype: type[pl.DataType] = pl.Float64,
    ):
        """Construct a CustomDataLoader instance.

        Primarily, this class is used to load one-point spectra data and coherence data from files
        and put them into a format that the CalibrationProblem class expects. The data is stored
        in a polars dataframe and then placed into a dictionary of DataFrames.

        Parameters
        ----------
        ops_data_file : Path | str
            Path to the one-point spectra data file.
        coherence_data_file : Path | str | None, optional
            Path to the coherence data file. If None, no coherence data will be loaded.
        dtype : pl.DataType, optional
            Data type to load the data in as.
        """
        # Set dtype
        self.dtype = dtype

        # Set data file paths
        self.ops_data_file = Path(ops_data_file)
        self.coherence_data_file = Path(coherence_data_file) if coherence_data_file else None

        # Load the OPS data
        self._load_ops_data()

        # Load the coherence data if provided
        if self.coherence_data_file:
            self._load_coherence_data()
        else:
            self.coh_data_df = None

    def _load_ops_data(self) -> None:
        """Load the OPS data into a polars dataframe.

        Must be able to find the following columns:
        - freq
        - uu
        - vv
        - ww
        - uw (optional)
        - vw (optional)
        - uv (optional)

        If the optional columns are not provided, they are set to NaN.
        """
        # Check the provided data file exists
        assert self.ops_data_file is not None, "Why isn't there an ops data file?"

        if not self.ops_data_file.exists():
            raise FileNotFoundError(f'Provided data file path "{self.ops_data_file}" does not exist.')

        # Load the data
        self.ops_data_df = pl.read_csv(self.ops_data_file)

        # Check that the required columns are present
        required_cols = ["freq", "uu", "vv", "ww"]
        for col in required_cols:
            if col not in self.ops_data_df.columns:
                raise ValueError(f"A(n) {col} column must be present in the ops_data_df")

        # Sort the data by frequency
        self.ops_data_df = self.ops_data_df.sort("freq")

        # Cast the data to the correct dtype
        self.ops_data_df = self.ops_data_df.with_columns(
            pl.col("freq").cast(self.dtype).alias("freq"),
            pl.col("uu").cast(self.dtype).alias("uu"),
            pl.col("vv").cast(self.dtype).alias("vv"),
            pl.col("ww").cast(self.dtype).alias("ww"),
        )

        # Check that the optional columns are present
        optional_cols = ["uw", "vw", "uv"]
        for col in optional_cols:
            if col not in self.ops_data_df.columns:
                self.ops_data_df = self.ops_data_df.with_columns(pl.lit(float("nan")).alias(col))
            self.ops_data_df = self.ops_data_df.with_columns(pl.col(col).cast(self.dtype))

    def _load_coherence_data(self) -> None:
        """Load the coherence data.

        Must be able to find the following columns:
        - r
        - freq
        - coh_u
        - coh_v
        - coh_w
        """
        # Check the provided data file exists
        assert self.coherence_data_file is not None, "Tried to load coherence data with no file?"

        if not self.coherence_data_file.exists():
            raise FileNotFoundError(f'Provided data file path "{self.coherence_data_file}" does not exist.')

        self.coh_data_df = pl.read_csv(self.coherence_data_file).sort("freq")

    def format_data(self) -> dict[str, pl.DataFrame | None]:
        """Format the data into a dictionary of tensors."""
        # Check that the data is loaded
        assert self.ops_data_df is not None, "OPS data not loaded??"
        if self.coh_data_df is None:
            print("No coherence data loaded.")

        return {
            "ops": self.ops_data_df,
            "coherence": self.coh_data_df,
        }
