from math import *
import numpy as np


####################################################################


class OnePointSpectraDataGenerator:
    def __init__(self, **kwargs):
        self.DataPoints = kwargs.get("DataPoints", None)
        self.flow_type = kwargs.get("flow_type", "shear")  # 'shear', 'iso'
        self.data_type = kwargs.get(
            "data_type", "Kaimal"
        )  # 'Kaimal', 'Simiu-Scanlan', 'Simiu-Yeo'

        if self.flow_type == "iso":
            self.eval = self.eval_iso
        elif self.flow_type == "shear":
            if self.data_type == "Kaimal":
                self.eval = self.eval_shear_Kaimal
            elif self.data_type == "Simiu-Scanlan":
                self.eval = self.eval_shear_SimiuScanlan
            elif self.data_type == "Simiu-Yeo":
                self.eval = self.eval_shear_SimiuYeo
            else:
                raise Exception()
        else:
            raise Exception()

        if self.DataPoints is not None:
            self.generate_Data(self.DataPoints)

    def generate_Data(self, DataPoints):
        DataValues = np.zeros([len(DataPoints), 3, 3])
        for i, Point in enumerate(DataPoints):
            DataValues[i] = self.eval(*Point)
        self.Data = (DataPoints, DataValues)
        return self.Data

    # =============================================
    # Models
    # =============================================

    ### TODO: correct spectra ? off-diagonal ?

    def eval_iso(self, k1, z=1):
        C = 3.2
        L = 0.59
        F = np.zeros([3, 3])
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

    def eval_shear_Kaimal(self, k1, z=1):
        n = 1 / (2 * pi) * k1 * z
        F = np.zeros([3, 3])
        F[0, 0] = 52.5 * n / (1 + 33 * n) ** (5 / 3)
        F[1, 1] = 8.5 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 1.05 * n / (1 + 5.3 * n ** (5 / 3))
        F[0, 2] = -7 * n / (1 + 9.6 * n) ** (2.4)
        return F

    def eval_shear_SimiuScanlan(self, k1, z=1):
        n = 1 / (2 * pi) * k1 * z
        F = np.zeros([3, 3])
        F[0, 0] = 100 * n / (1 + 50 * n) ** (5 / 3)
        F[1, 1] = 7.5 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 1.68 * n / (1 + 10 * n ** (5 / 3))
        return F

    def eval_shear_SimiuYeo(self, k1, z=1):
        n = 1 / (2 * pi) * k1 * z
        F = np.zeros([3, 3])
        F[0, 0] = 100 * n / (1 + 50 * n) ** (5 / 3)
        F[1, 1] = 7.5 * n / (1 + 10 * n) ** (5 / 3)
        F[2, 2] = 1.68 * n / (1 + 10 * n ** (5 / 3))
        return F


####################################################################
