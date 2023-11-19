"""Utilities for interpolating spectra provided in .csv format."""
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def extract_x_spectra(
    filepath: Path, abs: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a filepath to a csv with data in two cols; the first should
    be the x coordinates and the second should be the spectra value

    Parameters
    ----------
    filepath : Path
       Filepath to CSV

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of the x-coords on log scale and the spectra data
    """
    x, spectra = [], []
    with open(filepath) as spectra_csv:
        csv_reader = csv.reader(spectra_csv, delimiter=",")
        for row in csv_reader:
            x.append(float(row[0]))
            spectra.append(float(row[1]))

    spectra = np.array(spectra)
    if abs:
        spectra = np.abs(spectra)

    return np.log10(np.array(x)), np.array(spectra)


def interp_spectra(
    x_interp: np.ndarray, x_true: np.ndarray, spectra: np.ndarray
) -> np.ndarray:
    """
    Cubic spline interpolation of spectra over x_interp, given original
    x-coords x_true

    Parameters
    ----------
    x_interp : np.ndarray
        x-coords to evaluate the interpolation at
    x_true : np.ndarray
        x-coords to use in reaching the interpolation
    spectra : np.ndarray
        spectra date corresponding to the x-coords in x_true

    Returns
    -------
    np.ndarray
        The evaluation of the interpolation of the data spectra at the coords x_interp
    """
    cs = CubicSpline(x_true, spectra)

    return cs(x_interp)


def interpolate(
    datapath: Path, num_k1_points: int, plot: bool = False
) -> tuple[np.ndarray, ...]:
    """Calculates and returns the interpolations over a common set of x-coordinates.

    Parameters
    ----------
    datapath : Path
        Path to directory in which u_spectra.csv, v_spectra.csv, w_spectra.csv, uw_cospectra.csv are contained.
    num_k1_points : int
        Number of points to use in interpolating given spectra.
    plot : bool
        If True, will plot the interpolations after. Else just returns.

    Returns
    -------
    tuple[np.ndarray, ...]
        Tuple of x_interp, interp_u, interp_v, interp_w, interp_uw. x_interp is given in normal space, under the assumption that the x data of the input spectra is in log-space.
    """
    x_coords_u, u_spectra = extract_x_spectra(datapath / "u_spectra.csv")
    x_coords_v, v_spectra = extract_x_spectra(datapath / "v_spectra.csv")
    x_coords_w, w_spectra = extract_x_spectra(datapath / "w_spectra.csv")
    x_coords_uw, uw_cospectra = extract_x_spectra(
        datapath / "uw_cospectra.csv", abs=True
    )

    x_interp = np.linspace(min(x_coords_w), max(x_coords_w), num_k1_points)

    interp_u = interp_spectra(x_interp, x_coords_u, u_spectra)
    interp_v = interp_spectra(x_interp, x_coords_v, v_spectra)
    interp_w = interp_spectra(x_interp, x_coords_w, w_spectra)
    interp_uw = interp_spectra(x_interp, x_coords_uw, uw_cospectra)

    # The below will plot the data along with the cubic spline interpolation
    if plot:
        import matplotlib.pyplot as plt

        custom_palette = ["royalblue", "crimson", "forestgreen", "mediumorchid"]

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 8})

            plt.plot(
                x_coords_u,
                u_spectra,
                "o",
                label="Observed u Spectra",
                color=custom_palette[0],
            )
            plt.plot(x_interp, interp_u, color=custom_palette[0])

            plt.plot(
                x_coords_v,
                v_spectra,
                "o",
                label="Observed v Spectra",
                color=custom_palette[1],
            )
            plt.plot(x_interp, interp_v, color=custom_palette[1])

            plt.plot(
                x_coords_w,
                w_spectra,
                "o",
                label="Observed w Spectra",
                color=custom_palette[2],
            )
            plt.plot(x_interp, interp_w, color=custom_palette[2])

            plt.plot(
                x_coords_uw,
                uw_cospectra,
                "o",
                label="Observed uw Cospectra",
                color=custom_palette[3],
            )
            plt.plot(x_interp, interp_uw, color=custom_palette[3])

            plt.xlabel(r"$k_1$")
            plt.ylabel(r"$k_1 F_i /u_*^2$")
            plt.title("Logspace Spectra Interpolation")
            plt.legend()

        plt.show()

    return np.power(10, x_interp), interp_u, interp_v, interp_w, interp_uw
