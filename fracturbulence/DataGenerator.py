import warnings
from math import *

import numpy as np
import torch
from scipy.optimize import curve_fit, differential_evolution

####################################################################
#   One Point Spectra Data Generator
#   (iso, shear: Kaimal, Simiu-Scanlan, Simiu-Yeo)
####################################################################


class OnePointSpectraDataGenerator:
    def __init__(self, **kwargs):
        self.DataPoints = kwargs.get("DataPoints", None)
        self.data_type = kwargs.get(
            "data_type", "Kaimal"
        )  # 'Kaimal', 'Custom', 'Simiu-Scanlan', 'Simiu-Yeo'
        self.k1 = kwargs.get(
            "k1_data_points", None
        )  # TODO make this an optional, just like Datapoints and spectra_file

        self.zref = kwargs.get("zref", 1)
        self.Uref = kwargs.get("Uref", 1)

        if self.data_type == "VK":
            self.eval = self.eval_VK
        elif self.data_type == "Kaimal":
            self.eval = self.eval_Kaimal
        elif self.data_type == "IEC":
            self.eval = self.eval_IEC
        elif self.data_type == "Custom":
            if kwargs.get("spectra_file") is not None:
                print("Reading file" + kwargs.get("spectra_file") + "\n")
                spectra_file = kwargs.get("spectra_file")
                self.CustomData = torch.tensor(
                    np.genfromtxt(spectra_file, skip_header=1, delimiter=",")
                )
            else:
                raise Exception("Custom spectra_file not found")
        elif self.data_type == "Auto":
            if kwargs.get("spectra_file") is not None:
                print("Reading file" + kwargs.get("spectra_file") + "\n")
                spectra_file = kwargs.get("spectra_file")

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
                            minY = min(yData)

                            parameterBounds = []
                            parameterBounds.append([minX, maxX])  # search bounds for a
                            parameterBounds.append([minX, maxX])  # search bounds for b
                            parameterBounds.append(
                                [0.0, maxY]
                            )  # search bounds for Offset

                            # "seed" the numpy random number generator for repeatable results
                            result = differential_evolution(
                                sumOfSquaredError, parameterBounds, seed=3
                            )
                            return result.x

                        geneticParameters = generate_Initial_Parameters()

                        # curve fit the test data
                        fittedParameters, pcov = curve_fit(
                            func, xData, yData, geneticParameters, maxfev=50_000
                        )

                        print("Parameters", fittedParameters)

                        modelPredictions = func(xData, *fittedParameters)

                        absError = modelPredictions - yData

                        SE = np.square(absError)  # squared errors
                        MSE = np.mean(SE)  # mean squared errors
                        RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
                        Rsquared = 1.0 - (np.var(absError) / np.var(yData))
                        print("RMSE:", RMSE)
                        print("R-squared:", Rsquared)

                        return fittedParameters

                    Data_temp = np.genfromtxt(
                        spectra_file, skip_header=1, delimiter=","
                    )

                    DataValues = np.zeros([len(self.DataPoints), 3, 3])
                    fit1 = fitOPS(self.k1, Data_temp[:, 1], 1)
                    DataValues[:, 0, 0] = func124(self.k1, *fit1)
                    fit2 = fitOPS(self.k1, Data_temp[:, 2], 2)
                    DataValues[:, 1, 1] = func124(self.k1, *fit2)
                    fit3 = fitOPS(self.k1, Data_temp[:, 3], 4)
                    DataValues[:, 2, 2] = func124(self.k1, *fit3)
                    fit4 = fitOPS(self.k1, Data_temp[:, 4], 3)
                    DataValues[:, 0, 2] = -func3(self.k1, *fit4)

                    DataValues = torch.tensor(DataValues)

                    self.Data = (self.DataPoints, DataValues)
                    print(f"DataValues is on {DataValues.get_device()}")

                    self.CustomData = torch.tensor(Data_temp)
                else:
                    raise Exception("Custom k1 data not found")

            else:
                raise Exception("Custom spectra_file not found")

        else:
            raise Exception("No data type was provided")

        if self.DataPoints is not None and self.data_type is not "Auto":
            self.generate_Data(self.DataPoints)

    def compute_Fit(self, DataPoints, k1_data_points):
        pass

    def generate_Data(self, DataPoints):
        DataValues = torch.zeros([len(DataPoints), 3, 3])
        if self.data_type == "Custom":
            DataValues[:, 0, 0] = self.CustomData[:, 1]
            DataValues[:, 1, 1] = self.CustomData[:, 2]
            DataValues[:, 2, 2] = self.CustomData[:, 3]
            DataValues[:, 0, 2] = -self.CustomData[:, 4]
        else:
            for i, Point in enumerate(DataPoints):
                DataValues[i] = self.eval(*Point)

        print(f"DataValues is on {DataValues.get_device()}")

        self.Data = (DataPoints, DataValues)
        return self.Data

    def eval_VK(self, k1, z=1):
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

    def eval_Kaimal(self, k1, z=1):
        z = self.zref
        n = 1 / (2 * pi) * k1 * z
        F = torch.zeros([3, 3])
        F[0, 0] = 102 * n / (1 + 33 * n) ** (5 / 3)
        F[1, 1] = 17 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 2.1 * n / (1 + 5.3 * n ** (5 / 3))
        F[0, 2] = -12 * n / (1 + 9.6 * n) ** (7.0 / 3.0)
        return F

    def eval_auto(self, k1, z=1):

        pass

    def eval_IEC(self, k1, z=1):
        F = torch.zeros([3, 3])
        return F


####################################################################
#   One Point Spectra Data Generator
#   (iso, shear: Kaimal, Simiu-Scanlan, Simiu-Yeo)
####################################################################

"""
class OnePointSpectraDataGenerator:
    def __init__(self, **kwargs):
        self.DataPoints = kwargs.get("DataPoints", None)
        self.data_type = kwargs.get(
            "data_type", "Kaimal"
        )  # 'Kaimal', 'Custom', 'Simiu-Scanlan', 'Simiu-Yeo'
        self.k1 = kwargs.get('k1_data_points', None) # TODO make this an optional, just like Datapoints and spectra_file


        self.zref = kwargs.get("zref", 1)
        self.Uref = kwargs.get("Uref", 1)

        if self.data_type == "VK":
            self.eval = self.eval_VK
        elif self.data_type == "Kaimal":
            self.eval = self.eval_Kaimal
        elif self.data_type == "IEC":
            self.eval = self.eval_IEC
        elif self.data_type == "Custom":
            if kwargs.get("spectra_file") is not None:
                print("Reading file" + kwargs.get("spectra_file") + "\n")
                spectra_file = kwargs.get("spectra_file")
                self.CustomData = torch.tensor(
                    np.genfromtxt(spectra_file, skip_header=1, delimiter=",")
                )
            else:
                raise Exception("Custom spectra_file not found")
        elif self.data_type == 'Auto': 
            if kwargs.get('spectra_file') is not None:
                print('Reading file' + kwargs.get('spectra_file') + '\n')
                spectra_file=kwargs.get('spectra_file')

                if self.k1 is not None:                     
                    def func124(k1, a, b, c): 
                        ft = 1/(2*np.pi) * k1 * self.zref 

                        return a * ft / (1. + b*ft) ** c

                    def func3(k1, a, b, c): 
                        ft = 1/(2*np.pi) * k1 * self.zref 

                        return a * ft / (1 + b * ft ** c)

                    def fitOPS(xData, yData, num):
                        func = func3 if num == 3 else func124

                        def sumOfSquaredError(parameterTuple):
                            warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
                            val = func(xData, *parameterTuple)
                            return np.sum((yData - val) ** 2.0)

                        def generate_Initial_Parameters():
                            # min and max used for bounds
                            maxX = max(xData)
                            minX = min(xData)
                            maxY = max(yData)
                            minY = min(yData)

                            parameterBounds = []
                            parameterBounds.append([minX, maxX]) # search bounds for a
                            parameterBounds.append([minX, maxX]) # search bounds for b
                            parameterBounds.append([0.0, maxY]) # search bounds for Offset

                            # "seed" the numpy random number generator for repeatable results
                            result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
                            return result.x

                        geneticParameters = generate_Initial_Parameters()

                        # curve fit the test data
                        fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters, maxfev=50_000)

                        print('Parameters', fittedParameters)

                        modelPredictions = func(xData, *fittedParameters) 

                        absError = modelPredictions - yData

                        SE = np.square(absError) # squared errors
                        MSE = np.mean(SE) # mean squared errors
                        RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
                        Rsquared = 1.0 - (np.var(absError) / np.var(yData))
                        print('RMSE:', RMSE)
                        print('R-squared:', Rsquared)

                        return fittedParameters

                    Data_temp = np.genfromtxt(spectra_file,skip_header=1,delimiter=',')

                    DataValues = np.zeros([len(self.DataPoints), 3, 3])
                    fit1 = fitOPS(self.k1, Data_temp[:, 1], 1)
                    DataValues[:, 0, 0]  = func124(self.k1, *fit1)
                    fit2 = fitOPS(self.k1, Data_temp[:, 2], 2)
                    DataValues[:, 1, 1]  = func124(self.k1, *fit2)
                    fit3 = fitOPS(self.k1, Data_temp[:, 3], 4)
                    DataValues[:, 2, 2]  = func124(self.k1, *fit3)
                    fit4 = fitOPS(self.k1, Data_temp[:, 4], 3)
                    DataValues[:, 0, 2]  = -func3(self.k1, *fit4)

                    DataValues = torch.tensor(DataValues)

                    self.Data = (self.DataPoints, DataValues)
                    print(f"DataValues is on {DataValues.get_device()}")
                
                    self.CustomData=torch.tensor(Data_temp)
        else:
            raise Exception("No data type was provided")

        if self.DataPoints is not None:
            self.generate_Data(self.DataPoints)

    def generate_Data(self, DataPoints):
        DataValues = torch.zeros([len(DataPoints), 3, 3])
        if self.data_type == "Custom":
            DataValues[:, 0, 0] = self.CustomData[:, 1]
            DataValues[:, 1, 1] = self.CustomData[:, 2]
            DataValues[:, 2, 2] = self.CustomData[:, 3]
            DataValues[:, 0, 2] = -self.CustomData[:, 4]
        else:
            for i, Point in enumerate(DataPoints):
                DataValues[i] = self.eval(*Point)

        print(f"DataValues is on {DataValues.get_device()}")

        self.Data = (DataPoints, DataValues)
        return self.Data

    def eval_VK(self, k1, z=1):
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

    def eval_Kaimal(self, k1, z=1):
        z = self.zref
        n = 1 / (2 * pi) * k1 * z
        F = torch.zeros([3, 3])
        F[0, 0] = 102 * n / (1 + 33 * n) ** (5 / 3)
        F[1, 1] = 17 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 2.1 * n / (1 + 5.3 * n ** (5 / 3))
        F[0, 2] = -12 * n / (1 + 9.6 * n) ** (7.0 / 3.0)
        return F

    def eval_IEC(self, k1, z=1):
        F = torch.zeros([3, 3])
        return F
"""
