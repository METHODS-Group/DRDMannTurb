"""Generate one-point spectra data.

This module contains the ``OnePointSpectraDataGenerator`` class, which generates one-point spectra data for a given
set of parameters.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..enums import DataType


class OnePointSpectraDataGenerator:
    r"""One point spectra data generator.

    The one point spectra data generator, which evaluates one of a few spectral tensor models across a grid of
      :math:`k_1` wavevector points which is used as data in the fitting done in the :py:class:`OnePointSpectra` class.

    The type of spectral tensor is determined by the :py:enum:`DataType` argument, which determines one of the
    following models:

    #. ``DataType.KAIMAL``, which is the Kaimal spectra.

    #. ``DataType.VK``, which is the von Karman spectra model.

    #. ``DataType.CUSTOM``, usually used for data that is processed from real-world data. The spectra values are to be
        provided as the ``spectra_values`` field, or else to be loaded from a provided ``spectra_file``. The result is
        that the provided data are matched on the wavevector domain.

    #. ``DataType.AUTO``, which generates a filtered version of provided spectra data. The filtering is based on
        differential evolution to perform a non-linear fit onto functions of the following form:

    .. math::
        :nowrap:

        \begin{align}
            & \frac{k_1 F_{11}\left(k_1 z\right)}{u_*^2}=J_1(f):=\frac{a_1 f}{(1+b_1 f)^{c_1}} \\
            & \frac{k_1 F_{22}\left(k_1 z\right)}{u_*^2}=J_2(f):=\frac{a_2 f}{(1+b_2 f)^{c_2}} \\
            & \frac{k_1 F_{33}\left(k_1 z\right)}{u_*^2}=J_3(f):=\frac{a_3 f}{1+ b_3 f^{ c_3}} \\
            & -\frac{k_1 F_{13}\left(k_1 z\right)}{u_*^2}=J_4(f):=\frac{a_4 f}{(1+ b_4 f)^{c_4}},
        \end{align}

    with :math:`F_{12}=F_{23}=0`. Here, :math:`f = (2\pi)^{-1} k_1 z`. In the above, the :math:`a_i, b_i, c_i` are free
    parameters which are optimized by differential evolution. The result is a spectra model that is similar in form to
    the Kaimal spectra and which filters/smooths the spectra data from the real world and eases fitting by DRD models.
    This option is highly suggested in cases where spectra data have large deviations.

    .. note::
        The one-point spectra for :math:`F_{13}` are NOT to be pre-multiplied with a negative, this data generator
        automatically performs this step both when using ``DataType.CUSTOM`` and ``DataType.AUTO``.
    """

    def __init__(
        self,
        zref: float,
        ustar: float = 1.0,
        data_points: Optional[Sequence[tuple[torch.tensor, float]]] = None,
        data_type: DataType = DataType.KAIMAL,
        k1_data_points: Optional[torch.Tensor] = None,
        spectra_values: Optional[torch.Tensor] = None,
        spectra_file: Optional[Union[Path, str]] = None,
        seed: int = 3,
    ):
        r"""Initialize the OnePointSpectraDataGenerator.

        Parameters
        ----------
        zref : float
            Reference altitude value
        ustar : float
            Friction velocity, by default 1.0.
        data_points : Iterable[Tuple[torch.tensor, float]], optional
            Observed spectra data points at each of the :math:`k_1` coordinates, paired with the associated reference
            height (typically kept at 1, but may depend on applications).
        data_type : DataType, optional
            Indicates the data format to generate and operate with, by
            default ``DataType.KAIMAL``
        k1_data_points : Optional[Any], optional
            Wavevector domain of :math:`k_1`, by default None. This is only to be used when the ``AUTO`` tag is chosen
            to define the domain over which a non-linear regression is to be computed. See the interpolation module
            for examples of unifying different :math:`k_1` domains.
        spectra_file : Optional[Path], optional
            If using ``DataType.CUSTOM`` or ``DataType.AUTO``, this
            is used to indicate the data file (a .dat) to read
            from. Since it is not used by others, it is by
            default None

        Raises
        ------
        ValueError
            In the case that ``DataType.CUSTOM`` is indicated, but no spectra_file
            is provided
        ValueError
            In the case that ``DataType.AUTO`` data is indicated, but no k1_data_pts
            is provided
        ValueError
            Did not provide DataPoints during initialization for DataType method requiring spectra data.
        """
        self.DataPoints = data_points
        self.data_type = data_type
        self.k1 = k1_data_points

        self.seed = seed

        if spectra_values is not None:
            self.spectra_values = spectra_values

        self.zref = zref
        self.ustar = ustar

        if self.data_type == DataType.VK:
            self.eval = self.eval_VK

        elif self.data_type == DataType.KAIMAL:
            self.eval = self.eval_Kaimal

        elif self.data_type == DataType.CUSTOM:
            if spectra_file is not None:
                self.CustomData = torch.tensor(np.genfromtxt(spectra_file, skip_header=1, delimiter=","))
            else:
                raise ValueError("Indicated custom data type, but did not provide a spectra_file argument.")

        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

        return

    def generate_Data(
        self, DataPoints: Sequence[tuple[torch.tensor, float]]
    ) -> tuple[list[tuple[torch.Tensor, float]], torch.Tensor]:
        r"""Generate data from provided configuration.

        Generates a single spectral tensor from provided data or from a surrogate model.
        The resulting tensor is of shape (number of :math:`k_1` points):math:`\times 3 \times 3`,
        ie the result consists of the spectral tensor evaluated across the provided range of
        :math:`k_1` points. The spectra model is set during object instantiation.

        .. note::
            The ``DataType.CUSTOM`` type results in replication of the provided spectra data.

        Parameters
        ----------
        DataPoints : Iterable[Tuple[torch.tensor, float]]
            Observed spectra data points at each of the :math:`k_1` coordinates, paired with the associated reference
            height (typically kept at 1, but may depend on applications).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Evaluated spectral tensor on each of the provided grid points depending on the ``DataType`` selected.

        Raises
        ------
        ValueError
            DataType set such that an iterable set of spectra values is required. This is for any DataType other than
            ``CUSTOM`` and ``AUTO``.
        """
        DataValues = torch.zeros([len(DataPoints), 3, 3])

        if self.data_type == DataType.CUSTOM:
            DataValues[:, 0, 0] = self.CustomData[:, 1]  # uu
            DataValues[:, 1, 1] = self.CustomData[:, 2]  # vv
            DataValues[:, 2, 2] = self.CustomData[:, 3]  # ww
            # NOTE: is always negative
            DataValues[:, 0, 2] = -self.CustomData[:, 4]  # uw
            # TODO: Can be negative, skipping for now
            DataValues[:, 1, 2] = self.CustomData[:, 5]  # vw
            # TODO: Can be negative, skipping for now
            DataValues[:, 0, 1] = self.CustomData[:, 6]  # uv
        else:
            for i, Point in enumerate(DataPoints):
                DataValues[i] = self.eval(Point)

        DataPoints = list(zip(DataPoints, [self.zref] * len(DataPoints)))
        self.Data = (DataPoints, DataValues)
        return self.Data

    def eval_VK(self, k1: Union[torch.Tensor, float]) -> torch.Tensor:
        r"""Evaluate frequerncy-weighted von Karman spectral tensor.

        The von Karman spectral tensor is given by

        .. math::
                \Phi_{i j}^{\mathrm{VK}}(\boldsymbol{k})=\frac{E(k)}{4 \pi k^2}
                \left(\delta_{i j}-\frac{k_i k_j}{k^2}\right)

        which utilizes the energy spectrum function

        .. math::
            E(k)=c_0^2 \varepsilon^{2 / 3} k^{-5 / 3}\left(\frac{k L}{\left(1+(k L)^2\right)^{1 / 2}}\right)^{17 / 3},

        where :math:`\varepsilon` is the viscous dissipation of the turbulent kinetic energy, :math:`L` is the length
        scale parameter and :math:`c_0^2 \approx 1.7` is an empirical constant. Here, the physical constants are taken
        to be :math:``L = 0.59``.

        Parameters
        ----------
        k1 : torch.Tensor
            First dimension of the wavevector :math:`k_1`, this is a single coordinate of the domain over which the
            associated spectra are defined.

        Returns
        -------
        torch.Tensor
            The :math:`3 \times 3` matrix with entries determined by the von Karman spectra.
        """
        C = 3.2
        L = 0.59
        F = torch.zeros([3, 3])
        F[0, 0] = 9 / 55 * C / (L ** (-2) + k1**2) ** (5 / 6)
        F[1, 1] = 3 / 110 * C * (3 * L ** (-2) + 8 * k1**2) / (L ** (-2) + k1**2) ** (11 / 6)
        F[2, 2] = 3 / 110 * C * (3 * L ** (-2) + 8 * k1**2) / (L ** (-2) + k1**2) ** (11 / 6)
        return k1 * F

    def eval_Kaimal(self, k1: Union[torch.Tensor, float]) -> torch.Tensor:
        r"""Evaluate frequency-weighted Kaimal one-point spectra.

        Evaluates the one-point spectra as proposed by `Kaimal et al <https://apps.dtic.mil/sti/tr/pdf/AD0748543.pdf>`__
        in 1972. Clasically motivated by measurements taken over a flat homogeneous terrain in Kansas, the one-point
        spectra were proposed as

        .. math::
            :nowrap:

            \begin{align}
                & \frac{k_1 F_{11}\left(k_1 z\right)}{u_*^2}=J_1(f):=\frac{52.5 f}{(1+33 f)^{5 / 3}} \\
                & \frac{k_1 F_{22}\left(k_1 z\right)}{u_*^2}=J_2(f):=\frac{8.5 f}{(1+9.5 f)^{5 / 3}} \\
                & \frac{k_1 F_{33}\left(k_1 z\right)}{u_*^2}=J_3(f):=\frac{1.05 f}{1+5.3 f^{5 / 3}} \\
                & -\frac{k_1 F_{13}\left(k_1 z\right)}{u_*^2}=J_4(f):=\frac{7 f}{(1+9.6 f)^{12 / 5}},
            \end{align}

        with :math:`F_{12}=F_{23}=0`. Here, :math:`f = (2\pi)^{-1} k_1 z`. This method returns a :math:`3\times 3`
        matrix whose entries are determined by the above equations.


        Parameters
        ----------
        k1 : torch.Tensor
            First dimension of the wavevector :math:`k_1`, this is the domain over which the associated spectra
            are defined.

        Returns
        -------
        torch.Tensor
            The :math:`3 \times 3` matrix with entries determined by the Kaimal one-point spectra.
        """
        n = 1 / (2 * np.pi) * k1 * self.zref
        F = torch.zeros([3, 3])
        F[0, 0] = 52.5 * n / (1 + 33 * n) ** (5 / 3)
        F[1, 1] = 8.5 * n / (1 + 9.5 * n) ** (5 / 3)
        F[2, 2] = 1.05 * n / (1 + 5.3 * n ** (5 / 3))
        F[0, 2] = -7 * n / (1 + 9.6 * n) ** (12.0 / 5.0)

        return F * self.ustar**2
