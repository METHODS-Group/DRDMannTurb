from math import *
import numpy as np
import torch


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

        self.zref = kwargs.get("zref", 1)
        self.Uref = kwargs.get("Uref", 1)

        if self.data_type == "VK":
            self.eval = self.eval_VK

        elif self.data_type == "Kaimal":
            self.eval = self.eval_Kaimal

        elif self.data_type == "IEC":
            self.eval = self.eval_IEC
        elif self.data_type == 'Custom':
            if kwargs.get('spectra_file') is not None:
                print('Reading file' + kwargs.get('spectra_file') + '\n')
                spectra_file=kwargs.get('spectra_file')
                self.CustomData=torch.tensor(np.genfromtxt(spectra_file,skip_header=1,delimiter=','))
                # TODO: self.CustomData=torch.genfromtxt(spectra_file,skip_header=1,delimiter=',')

            else:
                raise Exception("Custom spectra_file not found")
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
