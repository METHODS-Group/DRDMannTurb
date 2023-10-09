"""Utilities for interpolating spectra provided in .csv format."""
import csv

import numpy as np
from scipy.interpolate import CubicSpline


def extract_x_spectra(filepath):
    """_summary_

    Parameters
    ----------
    filepath : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    x, spectra = [], []
    with open(filepath) as spectra_csv:
        csv_reader = csv.reader(spectra_csv, delimiter=",")
        for row in csv_reader:
            x.append(float(row[0]))
            spectra.append(float(row[1]))

    return np.log10(np.array(x)), np.array(spectra)


def interp_spectra(x_interp, x_true, spectra):
    """Cubic Spline interpolation of spectra on given discretization and returned on a newly requested discretization.

    Parameters
    ----------
    x_interp : _type_
        _description_
    x_true : _type_
        _description_
    spectra : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    cs = CubicSpline(x_true, spectra)

    return cs(x_interp)
