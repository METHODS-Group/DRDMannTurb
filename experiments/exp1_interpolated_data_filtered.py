import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline

from drdmannturb.calibration.calibration import CalibrationProblem
from drdmannturb.common import MannEddyLifetime
from drdmannturb.calibration.data_generator import OnePointSpectraDataGenerator

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
    "nepochs": 2,
    "curves": [0, 1, 2, 3],
    "data_type": "Auto",
    "spectra_file": "data/Spectra_interp.dat",
    "Uref": 21,
    # "zref": 1,
    "zref": 80,
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
SIGMA = 0.04
# SIGMA = 3.2
# UREF = 21

"""
Define the File IO routines
"""


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

    logscale_x = np.log10(np.array(x))

    # return np.log10(np.array(x)), np.array(spectra)
    return logscale_x, spectra


# TODO -- x-axis is different from the others -- frequency multiplied ... eg, different units
# TODO -- note also possibly same discrepancy with y-axis/ need to check whether or not unit


def export_interpolation(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    uw: np.ndarray,
    filename: str = "Spectra_interp",
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

    print("Wrote file...")


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
        Path().resolve() / "data" / "uw_cospectra.csv", True
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

    return np.power(10, x_interp), interp_u, interp_v, interp_w, interp_uw


"""
Define the driving functions
"""


def driver(plot_loss: bool, plot_result: bool) -> None:
    """
    Driving function

    Parameters
    ----------
    plot_result : bool
        If true, plots loss against epoch #
    """
    config = CONSTANTS_CONFIG
    config["activations"] = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
    config["hlayers"] = [32] * 4

    pb = CalibrationProblem(**config)
    parameters = pb.parameters
    parameters[:3] = [np.log(L), np.log(GAMMA), np.log(SIGMA)]

    pb.parameters = parameters[: len(pb.parameters)]

    k1_data_pts = config["domain"]
    spectra_file = config["spectra_file"]
    print(f"READING FILE {spectra_file}\n\n")
    CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))
    f = CustomData[:, 0]
    k1_data_pts = 2 * torch.pi * f / Uref

    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(
        DataPoints=DataPoints, k1_data_points=k1_data_pts.data.numpy(), **config
    ).Data

    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=Data, **config)

    if plot_loss:
        plt.figure()

        plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
        plt.legend()
        plt.xlabel("Epoch Number")
        plt.ylabel("MSE")
        # plt.yscale("log")

        if not plot_result:
            plt.show()

    if plot_result:
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
        "-pL",
        "--plot-loss",
        action="store_true",
        help="Plots the resulting loss against epoch # graph",
    )
    parser.add_argument(
        "-pR",
        "--plot-result",
        action="store_true",
        help="Plots the result",
    )
    parser.add_argument(
        "-eI",
        "--export-interp",
        action="store_true",
        help="Writes out a file Spectra_interp",
    )
    parser.add_argument(
        "-b",
        "--beta-penal",
        type=float,
        default=0.0,
        help="Provide a coefficient for the additional penalization term",
    )
    parser.add_argument(
        "-p",
        "--penal",
        type=float,
        default=1.0,
        help="Provide a coefficient for the additional penalization term",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="Number of epochs to run"
    )

    args = parser.parse_args()
    if args.plot_interp:
        print("Will plot interp")
    if args.plot_loss:
        print("Will plot loss")

    CONSTANTS_CONFIG["beta_penalty"] = args.beta_penal
    CONSTANTS_CONFIG["penalty"] = args.penal
    CONSTANTS_CONFIG["nepochs"] = args.epochs

    x_interp, interp_u, interp_v, interp_w, interp_uw = interpolate(args.plot_interp)
    if args.export_interp:
        export_interpolation(
            x_interp, interp_u, interp_v, interp_w, interp_uw, "data/Spectra_interp"
        )

    # NOTE: update the config to the problem from above
    CONSTANTS_CONFIG["domain"] = torch.from_numpy(x_interp)

    driver(args.plot_loss, args.plot_result)
