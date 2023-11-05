import warnings
from pathlib import Path
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit, differential_evolution

from drdmannturb.enums import DataType


class OnePointSpectraDataGenerator:
    """
    One point spectra data generator

    TODO -- better introduction/ description of its purpose in the code
    """

    def __init__(
        self,
        data_points: Optional[Any] = None,
        data_type: DataType = DataType.KAIMAL,
        k1_data_points: Optional[Any] = None,  # TODO: properly type
        spectra_values: Optional[Any] = None,  # TODO: properly type
        spectra_file: Optional[Path] = None,
        zref: float = 1.0,
        seed: int = 3,
    ):
        """
        TODO -- This needs a better introduction

        Parameters
        ----------
        data_points : Optional[Any], optional
            _description_, by default None
        data_type : DataType, optional
            Indicates the data format to generate and operate with, by
            default DataType.KAIMAL
        k1_data_points : Optional[Any], optional
            _description_, by default None
        spectra_file : Optional[Path], optional
            If using DataType.CUSTOM or DataType.AUTO, this
            is used to indicate the data file (a .dat) to read
            from. Since it is not used by others, it is by
            default None
        zref : float, optional
            Reference altitude value, by default 1.0

        Raises
        ------
        ValueError
            In the case that DataType.CUSTOM is indicated, but no spectra_file
            is provided
        ValueError
            In the case that DataType.AUTO data is indicated, but no k1_data_pts
            is provided
        """
        # TODO -- Note the Any annotations above; need to figure out what type should be given

        self.DataPoints = data_points
        self.data_type = data_type
        self.k1 = k1_data_points

        self.seed = seed

        if spectra_values is not None:
            self.spectra_values = spectra_values

        self.zref = zref

        if self.data_type == DataType.VK:
            self.eval = self.eval_VK

        elif self.data_type == DataType.KAIMAL:
            self.eval = self.eval_Kaimal

        elif self.data_type == DataType.IEC:
            self.eval = self.eval_IEC

        elif self.data_type == DataType.CUSTOM:
            if spectra_file is not None:
                self.CustomData = torch.tensor(
                    np.genfromtxt(spectra_file, skip_header=1, delimiter=",")
                )
            else:
                raise ValueError(
                    "Indicated custom data type, but did not provide a " "spectra_file"
                )

        elif self.data_type == DataType.AUTO:
            if self.k1 is not None:

                def func124(k1, a, b, c):
                    ft = 1 / (2 * np.pi) * k1 * self.zref

                    return a * ft / (1.0 + b * ft) ** c

                def func3(k1, a, b, c):
                    ft = 1 / (2 * np.pi) * k1 * self.zref

                    return a * ft / (1 + b * ft**c)

                def fitOPS(xData, yData, num):
                    func = func3 if num == 3 else func124

                    def sumOfSquaredError(parameterTuple):
                        warnings.filterwarnings(
                            "ignore"
                        )  # do not print warnings by genetic algorithm
                        val = func(xData, *parameterTuple)
                        return np.sum((yData - val) ** 2.0)

                    def generate_Initial_Parameters():
                        # min and max used for bounds
                        maxX = max(xData)
                        minX = min(xData)
                        maxY = max(yData)

                        parameterBounds = []
                        # search bounds for a
                        parameterBounds.append([minX, maxX])
                        # search bounds for b
                        parameterBounds.append([minX, maxX])
                        parameterBounds.append([0.0, maxY])  # search bounds for Offset

                        # "seed" the numpy random number generator for
                        #   replicable results
                        result = differential_evolution(
                            sumOfSquaredError, parameterBounds, seed=seed
                        )
                        return result.x

                    geneticParameters = generate_Initial_Parameters()

                    # curve fit the test data
                    fittedParameters, _ = curve_fit(
                        func, xData, yData, geneticParameters, maxfev=50_000
                    )

                    modelPredictions = func(xData, *fittedParameters)

                    absError = modelPredictions - yData

                    SE = np.square(absError)  # squared errors
                    MSE = np.mean(SE)  # mean squared errors
                    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
                    Rsquared = 1.0 - (np.var(absError) / np.var(yData))

                    return fittedParameters

                if self.spectra_values is not None:
                    Data_temp = self.spectra_values.copy()
                else:
                    raise ValueError(
                        "Indicated DataType.AUTO, but did not provide raw spectra data. "
                    )

                DataValues = np.zeros([len(self.DataPoints), 3, 3])
                print("Filtering provided spectra interpolation.")
                print("=" * 30)

                fit1 = fitOPS(self.k1, Data_temp[:, 0], 1)
                DataValues[:, 0, 0] = func124(self.k1, *fit1)

                fit2 = fitOPS(self.k1, Data_temp[:, 1], 2)
                DataValues[:, 1, 1] = func124(self.k1, *fit2)

                fit3 = fitOPS(self.k1, Data_temp[:, 2], 4)
                DataValues[:, 2, 2] = func124(self.k1, *fit3)

                fit4 = fitOPS(self.k1, Data_temp[:, 3], 3)
                DataValues[:, 0, 2] = -func3(self.k1, *fit4)

                DataValues = torch.tensor(DataValues)

                self.Data = (self.DataPoints, DataValues)

                self.CustomData = torch.tensor(Data_temp)
            else:
                raise ValueError(
                    "Indicated DataType.AUTO, but did not provide k1_data_points"
                )

        if self.DataPoints is not None and self.data_type != DataType.AUTO:
            self.generate_Data(self.DataPoints)

        return

    def generate_Data(self, DataPoints) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO -- documentation

        Parameters
        ----------
        DataPoints : torch.Tensor
            _description_

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of the input DataPoints and generated DataValues
        """
        DataValues = torch.zeros([len(DataPoints), 3, 3])

        if self.data_type == DataType.CUSTOM:
            DataValues[:, 0, 0] = self.CustomData[:, 1]
            DataValues[:, 1, 1] = self.CustomData[:, 2]
            DataValues[:, 2, 2] = self.CustomData[:, 3]
            DataValues[:, 0, 2] = -self.CustomData[:, 4]

        else:
            # TODO -- there should be a way to vectorize this operation
            for i, Point in enumerate(DataPoints):
                DataValues[i] = self.eval(*Point)

        self.Data = (DataPoints, DataValues)
        return self.Data

    def eval_VK(self, k1: float, z: float = 1.0) -> torch.Tensor:
        """
        eval implementation for VK data type

        Parameters
        ----------
        k1 : torch.Tensor
            _description_
        z : int, optional
            _description_, by default 1

        Returns
        -------
        torch.Tensor
            Result of the evaluation
        """

        C = 3.2
        L = 0.59
        F = torch.zeros([3, 3])
        F[0, 0] = 9 / 55 * C / (L ** (-2) + k1**2) ** (5 / 6)
        F[1, 1] = (
            3
            / 110
            * C
            * (3 * L ** (-2) + 8 * k1**2)
            / (L ** (-2) + k1**2) ** (11 / 6)
        )
        F[2, 2] = (
            3
            / 110
            * C
            * (3 * L ** (-2) + 8 * k1**2)
            / (L ** (-2) + k1**2) ** (11 / 6)
        )
        return k1 * F

    def eval_Kaimal(self, k1: float, z: float = 1.0) -> torch.Tensor:
        """
        eval implementation for Kaimal data type

        Parameters
        ----------
        k1 : torch.Tensor
            _description_
        z : int, optional
            _description_, by default 1

        Returns
        -------
        torch.Tensor
            Result of the evaluation
        """
        z = self.zref
        n = 1 / (2 * np.pi) * k1 * z
        F = torch.zeros([3, 3])
        F[0, 0] = 102 * n / (1 + 33 * n) ** (5 / 3)
        F[1, 1] = 17 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 2.1 * n / (1 + 5.3 * n ** (5 / 3))
        F[0, 2] = -12 * n / (1 + 9.6 * n) ** (7.0 / 3.0)
        return F

    def eval_IEC(self, **_) -> torch.Tensor:
        """
        Eval implementation for IEC data type. Returns constant
        3x3 of zeros.

        Returns
        -------
        torch.Tensor
            Result of the evaluation; constant 3x3 zeros tensor
        """

        F = torch.zeros([3, 3])
        return F

    def plot(
        self,
        x_interp: Optional[np.ndarray] = None,
        spectra_full: Optional[np.ndarray] = None,
        x_coords_full: Optional[np.ndarray] = None,
    ):
        """_summary_

        Parameters
        ----------
        x_interp : Optional[np.ndarray], optional
            _description_, by default None
        spectra_full : Optional[np.ndarray], optional
            _description_, by default None
        x_coords_full : Optional[np.ndarray], optional
            _description_, by default None

        Raises
        ------
        ValueError
            Provide interpolation domain (normal space) for filtered plots.
        ValueError
            Provide original spectra and associated domains as a single stack of np.arrays, even if all x coordinates are matching.

        """

        cmap = plt.get_cmap("Spectral", 4)
        custom_palette = [mpl.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

        if self.data_type == DataType.AUTO:
            if x_interp is None:
                raise ValueError(
                    "Provide interpolation domain (normal space) for filtered plots."
                )

            if spectra_full is None or x_coords_full is None:
                raise ValueError(
                    "Provide original spectra and associated domains as a single stack of np.arrays, even if all x coordinates are matching."
                )

            x_interp_plt = np.log10(x_interp)
            filtered_data_fit = (
                self.Data[1].cpu().numpy()
                if torch.cuda.is_available()
                else self.Data[1].numpy()
            )

            with plt.style.context("bmh"):
                fig, ax = plt.subplots()

                ax.plot(
                    x_interp_plt,
                    filtered_data_fit[:, 0, 0],
                    label="Filtered u spectra",
                    color=custom_palette[0],
                )

                ax.plot(
                    x_coords_full[0],
                    spectra_full[0],
                    "o",
                    label="Observed u spectra",
                    color=custom_palette[0],
                )

                ax.plot(
                    x_interp_plt,
                    filtered_data_fit[:, 1, 1],
                    label="Filtered v spectra",
                    color=custom_palette[1],
                )

                ax.plot(
                    x_coords_full[1],
                    spectra_full[1],
                    "o",
                    label="Observed v spectra",
                    color=custom_palette[1],
                )

                ax.plot(
                    x_interp_plt,
                    filtered_data_fit[:, 2, 2],
                    label="Filtered w spectra",
                    color=custom_palette[2],
                )

                ax.plot(
                    x_coords_full[2],
                    spectra_full[2],
                    "o",
                    label="Observed w spectra",
                    color=custom_palette[2],
                )

                ax.plot(
                    x_interp_plt,
                    filtered_data_fit[:, 0, 2],
                    label="Filtered uw cospectra",
                    color=custom_palette[3],
                )

                ax.plot(
                    x_coords_full[3],
                    spectra_full[3],
                    "o",
                    label="Observed uw cospectra",
                    color=custom_palette[3],
                )

                ax.set_title("Filtered and Original Spectra")
                ax.set_xlabel(r"$k_1$")
                ax.set_ylabel(r"$k_1 F_i /u_*^2$")
                ax.legend()

                fig.show()
