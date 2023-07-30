import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from fracturbulence.Calibration import CalibrationProblem
from fracturbulence.common import MannEddyLifetime
from fracturbulence.DataGenerator import OnePointSpectraDataGenerator
from scipy.interpolate import CubicSpline
from typing import Optional

plt.style.use("bmh")

"""
The below segment defines all constants required below
"""

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

savedir = Path(__file__).parent / "results"

CONSTANTS_CONFIG = {
    "type_EddyLifetime": "customMLP",  # CALIBRATION : 'tauNet',  ### 'const', TwoThird', 'Mann', 'customMLP', 'tauNet'
    "type_PowerSpectra": "RDT",  # 'RDT', 'zetaNet', 'C3Net', 'Corrector'
    "learn_nu": False,  # NOTE: Experiment 1: False, Experiment 2: True
    "plt_tau": True,
    "hlayers": [10, 10],  # ONLY NEEDED FOR CUSTOMNET OR RESNET
    "tol": 1.0e-9,  # not important
    "lr": 1,  # learning rate
    "penalty": 1,  # CALIBRATION: 1.e-1,
    "regularization": 1.0e-5,  # CALIBRATION: 1.e-1,
    "nepochs": 10,
    "curves": [0, 1, 2, 3],
    "data_type": "Custom",  # CALIBRATION: 'Custom', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    # "data_type": "Auto",  # CALIBRATION: 'Custom', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    "spectra_file": "Spectra_exp.dat",
    "Uref": 10,  # m/s
    "zref": 1,  # m
    "domain": None,
    "noisy_data": 0.0,  # 0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
}

zref = CONSTANTS_CONFIG["zref"]
# Hub height in meters
Uref = CONSTANTS_CONFIG["Uref"]
# Average Hub height velocity in m/s
Iref = 0.14
sigma1 = Iref * (0.75 * Uref + 5.6)
Lambda1 = 42
# Longitudinal turbulence scale parameter at hub height

z0 = 0.01
ustar = 0.41 * Uref / np.log(zref / z0)

# NOTE: values taken from experiment1 in the paper
L = 0.59
GAMMA = 3.9
SIGMA = 3.2


"""
The below segment plots the spectra data interpolated over a
common set of x coordinates
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

    return np.log10(np.array(x)), np.array(spectra)


def export_into_spectra(
    x: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    uw: np.ndarray,
    filename: Optional[str] = "Spectra_exp",
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

    with open(filename, 'w') as csvfile:
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


def driver_interp(plot_interp: bool):
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
        # cs = CubicSpline(x_coords_w, w_spectra)
        # y_interp = cs(x_interp)
        # y_cs_true = cs(x_coords_u)
        # cmap = mpl.cm.get_cmap('Spectral')
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

        # mse_interp = mean_squared_error(y_cs_true, w_spectra)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.title(f"MSE of Interpolant = {mse_interp}")
        plt.title("Spectra Interpolation")
        plt.legend()
        plt.show()

    return x_interp, interp_u, interp_v, interp_w, interp_uw


def driver(x_interp, plot_result=False):
    """
    Driving function
    """

    # Set up config
    CONSTANTS_CONFIG["domain"] = torch.from_numpy(x_interp)

    # Create calibration problem
    pb = CalibrationProblem(**CONSTANTS_CONFIG)

    # Set up pb.parameters
    parameters = pb.parameters
    parameters[:3] = [
        np.log(L),
        np.log(GAMMA),
        np.log(SIGMA),
    ]  # All of these parameters are positive
    # so we can train the NN for the log of these parameters.
    pb.parameters = parameters[:len(pb.parameters)]

    k1_data_pts = CONSTANTS_CONFIG["domain"]
    DataPoints = [(k1, 1) for k1 in k1_data_pts]
    Data = OnePointSpectraDataGenerator(DataPoints=DataPoints, k1_data_points=DataPoints, **CONSTANTS_CONFIG).CustomData

    DataValues = Data[1]

    IECtau = MannEddyLifetime(k1_data_pts * L)
    kF = pb.eval(k1_data_pts)

    opt_params = pb.calibrate(Data=(Data, DataValues), **CONSTANTS_CONFIG)

    if plot_result:
        plt.plot(pb.loss_history_epochs, "o-", label="Epochs Loss History")
        plt.legend()
        plt.xlabel("Epoch Number")
        plt.ylabel("MSE")
        plt.yscale("log")


if __name__ == "__main__":
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

    print("Interpolating")
    out_tuple = driver_interp(args.plot_interp)
    export_into_spectra(*out_tuple)
    print("Running through the RDT NN")
    driver(out_tuple[0], plot_result=args.plot_result)
