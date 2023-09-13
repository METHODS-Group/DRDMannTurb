"""
NOTE: this module is not used throughout the code and is vestigial from
earlier versions. It is not maintained either.
"""

"""An inlet boundary condition process for KratosMultiphysics

license: license.txt
"""

__all__ = ["Factory", "ImposeWindInletProcess"]

from collections import Mapping, namedtuple


# from math import isclose
def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


from math import ceil, floor, log
from time import time

import h5py
import KratosMultiphysics
import matplotlib.pyplot as plt
import numpy as np
from CovarianceKernels import MannCovariance, VonKarmanCovariance
from GaussianRandomField import *
from GenerateWind import GenerateWind
from KratosMultiphysics import (
    DELTA_TIME,
    TIME,
    VELOCITY_X,
    VELOCITY_Y,
    VELOCITY_Z,
    Logger,
)

##DONE: added support for power law
##DONE: read all mean profile parameters from json file
##DONE: can choose between power law and logarithmic profile


class Parameters(Mapping):
    def __init__(self, kratos_parameters):
        self._kratos_parameters = kratos_parameters

    def __getitem__(self, key):
        param = self._kratos_parameters[key]
        if param.IsDouble():
            value = param.GetDouble()
        elif param.IsInt():
            value = param.GetInt()
        elif param.IsString():
            value = param.GetString()
        else:
            value = param
        return value

    def __iter__(self):
        yield from self._kratos_parameters.keys()

    def __len__(self):
        return self._kratos_parameters.size()


def Factory(settings, Model):
    return ImposeWindInletProcess(Model, Parameters(settings["Parameters"]))


class LogMeanProfile:
    def __init__(self, friction_velocity, roughness_height, bulk_wind_speed, dim):
        self.friction_velocity = friction_velocity
        self.roughness_height = roughness_height
        self.bulk_wind_speed = bulk_wind_speed
        self.dim = dim

    def get_height(self, node):
        if self.dim == 2:
            height = node.Y
        elif self.dim == 3:
            height = node.Z
        return height

    def wind_speed(self, node):
        return (
            self.friction_velocity
            / 0.41
            * log(
                (self.get_height(node) + self.roughness_height) / self.roughness_height
            )
        )


class ImposeWindInletProcess:
    @property
    def inlet_nodes(self):
        return self.model_part.Nodes

    def __init__(self, Model, settings):
        for name, value in settings.items():
            setattr(self, name, value)
        self.model_part = Model[self.inlet_model_part_name]
        self.mean_profile = self.CreateMeanProfile()

    def ExecuteInitialize(self):
        for node in self.inlet_nodes:
            for var in [VELOCITY_X, VELOCITY_Y, VELOCITY_Z]:
                node.Fix(var)

    def ExecuteInitializeSolutionStep(self):
        self.AssignVelocity()
        self.ApplyRamp()

    def CreateMeanProfile(self, dim=3):
        reference_height = self.reference_height
        # lz = self.lz
        umean = self.umean
        roughness_height = self.roughness_height
        self.bulk_wind_speed = umean
        # self.bulk_wind_speed = umean * (log(lz/roughness_height) - 1.0) / log(reference_height/roughness_height)
        self.friction_velocity = (
            umean * 0.41 / log((reference_height + roughness_height) / roughness_height)
        )
        return LogMeanProfile(
            self.friction_velocity, roughness_height, self.bulk_wind_speed, dim
        )

    def AssignVelocity(self):
        for var in [VELOCITY_X, VELOCITY_Y, VELOCITY_Z]:
            if var == VELOCITY_X:
                for node in self.inlet_nodes:
                    vel = self.mean_profile.wind_speed(node)
                    node.SetSolutionStepValue(var, vel)
            else:
                for node in self.inlet_nodes:
                    vel = 0.0
                    node.SetSolutionStepValue(var, vel)

    def ApplyRamp(self):
        time = self.model_part.ProcessInfo[TIME]
        if time < self.ramp_time:
            scal = time / self.ramp_time
            for node in self.inlet_nodes:
                for var in [VELOCITY_X, VELOCITY_Y, VELOCITY_Z]:
                    vel = node.GetSolutionStepValue(var)
                    node.SetSolutionStepValue(var, scal * vel)

    def Check(self):
        pass

    def ExecuteBeforeSolutionLoop(self):
        pass

    def ExecuteFinalizeSolutionStep(self):
        pass

    def ExecuteBeforeOutputStep(self):
        pass

    def ExecuteAfterOutputStep(self):
        pass

    def ExecuteFinalize(self):
        pass
