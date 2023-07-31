import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import MannEddyLifetime
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator
from scipy.interpolate import CubicSpline
from typing import Optional

plt.style.use("bmh")

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

"""
Set constants, etc.
"""

savedir = Path(__file__).parent / "results"

CONSTANTS_CONFIG = {
    "type_EddyLifetime": "customMLP",
    "type_PowerSpectra": "RDT",
    "learn_nu": False,
    "plt_tau": True,
    "hlayers": [10, 10],
    "tol": 1.0e-9,
    "lr": 1,
    "penalty": 1,
    "regularization": 1.0e-5,
    "nepochs": 5,
    "curves": [0, 1, 2, 3],
    "data_type": "Kaimal",
    # "spectra_file": "constants/Spectra.dat",
    "spectra_file": "data/Spectra_interp.dat",
    # "Uref": 10,
    "Uref": 21,
    "zref": 1,
    "domain": torch.logspace(
        -1, 2, 20
    ),  # NOTE: This gets updated in the script portion
    "noisy_data": 0.0,
    "output_folder": str(savedir),
}

zref = CONSTANTS_CONFIG["zref"]
Uref = CONSTANTS_CONFIG["Uref"]
Iref = 0.14
sigma1 = Iref * (0.75 * Uref + 5.6)
Lambda1 = 42

z0 = 0.01
ustar = 0.41 * Uref / np.log(zref / z0)

# L = 0.59
# GAMMA = 3.9
# SIGMA = 3.2

L = 70
GAMMA = 3.7
SIGMA = 3.2
# UREF = 21

"""
Define the File IO routines
"""


def extract_x_spectra(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
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

    # return np.log10(np.array(x)), np.array(spectra)
    return np.array(x), np.array(spectra)


def export_into_spectra(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    uw: np.ndarray,
    filename: str = "Spectra_exp",
) -> None:
    """
    Takes the inputs, which should each be interpreted as columns to be
    written into the CSV

    Parameters
    ----------
    x : np.ndarray
        X-coords; f
    u : np.ndarray
        U-spectra
    v : np.ndarray
        V-spectra
    w : np.ndarray
        W-spectra
    uw : np.ndarray
        UW
    filename : str, optional
        The filename to write out to, by default "Spectra_exp"
    """
    contents = (np.vstack((x, u, v, w, uw))).transpose()

    filename = filename + ".dat"

    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(contents)


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


"""
Given the following data files, the following driving function will return
"""


def interpolate(plot_interp: bool) -> tuple[np.ndarray, ...]:
    """
    Calculates and returns the interpolations over a common set of x-coordinates

    Parameters
    ----------
    plot_interp : bool
        If True, will plot the interpolations after. Else just returns.

    Returns
    -------
    tuple[np.ndarray, ...]
        Tuple of x_interp, interp_u, interp_v, interp_w, interp_uw
    """
    x_coords_u, u_spectra = extract_x_spectra(
        Path().resolve() / "data" / "u_spectra.csv"
    )
    x_coords_v, v_spectra = extract_x_spectra(
        Path().resolve() / "data" / "v_spectra.csv"
    )
    x_coords_w, w_spectra = extract_x_spectra(
        Path().resolve() / "data" / "w_spectra.csv"
    )
    x_coords_uw, uw_cospectra = extract_x_spectra(
        Path().resolve() / "data" / "uw_cospectra.csv"
    )

    x_interp = np.linspace(min(x_coords_w), max(x_coords_w), 40)
    # x_interp = np.linspace(-2, 2, 20)

    # y_interp = barycentric_interpolate(x_coords_w, w_spectra, x_interp)
    interp_u = interp_spectra(x_interp, x_coords_u, u_spectra)
    interp_v = interp_spectra(x_interp, x_coords_v, v_spectra)
    interp_w = interp_spectra(x_interp, x_coords_w, w_spectra)
    interp_uw = interp_spectra(x_interp, x_coords_uw, uw_cospectra)

    # The below will plot the data along with the cubic spline interpolation
    if plot_interp:
        print("Plotting interpolations of experimental data")

        cmap = plt.get_cmap("Spectral", 4)
        custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

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

        plt.title("Spectra Interpolation")
        plt.legend()
        plt.show()

    return x_interp, interp_u, interp_v, interp_w, interp_uw


"""
Define the driving functions
"""


def driver(plot_result: bool) -> None:
    """
    Driving function

    Parameters
    ----------
    plot_result : bool
        If true, plots results
    """
    config = CONSTANTS_CONFIG
    config["activations"] = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
    config["hlayers"] = [32] * 4

    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [np.log(L), np.log(GAMMA), np.log(SIGMA)]

    pb.parameters = parameters[: len(pb.parameters)]
    k1_data_pts = config["domain"]
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, **config).Data

    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)

    if plot_result:
        plt.figure()

        plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
        plt.legend()
        plt.xlabel("Epoch Number")
        plt.ylabel("MSE")
        plt.yscale("log")

        plt.show()


if __name__ == "__main__":
    """
    Script segment
    """

    parser = argparse.ArgumentParser(
        description="experimenting with the interpolations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-pI",
        "--plot-interp",
        action="store_true",
        help="Plots the interpolated spectra data as is",
    )
    parser.add_argument(
        "-pR",
        "--plot-result",
        action="store_true",
        help="Plots the resulting data fit",
    )

    args = parser.parse_args()
    if args.plot_interp:
        print("Will plot interp")
    if args.plot_result:
        print("Will plot result")

    x_interp, interp_u, interp_v, interp_w, interp_uw = interpolate(args.plot_interp)

    # NOTE: update the config to the problem from above
    CONSTANTS_CONFIG["domain"] = torch.from_numpy(x_interp)

    driver(args.plot_result)
