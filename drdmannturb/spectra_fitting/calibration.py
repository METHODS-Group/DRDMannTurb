"""
This module implements the exposed CalibrationProblem class.
"""

import os
import pickle
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from ..common import MannEddyLifetime
from ..enums import EddyLifetimeType
from ..parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from .loss_functions import LossAggregator
from .one_point_spectra import OnePointSpectra

tqdm = partial(tqdm, position=0, leave=True)


class CalibrationProblem:
    r"""
    .. _calibration-problem-reference:

    Class which manages the spectra fitting and eddy lifetime function learning based on the deep rapid distortion model developed
    in `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_.

    This class manages the operator regression task which characterizes the best fitting candidate
    in a family of nonlocal covariance kernels that are parametrized by a neural network.

    This class can also be used independently of neural networks via the ``EddyLifetimeType`` used for classical spectra fitting tasks, for instance, using
    the ``EddyLifetimeType.MANN`` results in a fit that completely relies on the Mann eddy lifetime function. If a neural network model is used, Torch's ``LBFGS`` optimizer is used with cosine annealing for learning rate scheduling. Parameters for these components of the training process are set in ``LossParameters`` and ``ProblemParameters`` during initialization.

    After instantiating ``CalibrationProblem``, wherein the problem and eddy lifetime function substitution
    type are indicated, the user will need to generate the OPS data using ``OnePointSpectraDataGenerator``,
    after which the model can be fit with ``CalibrationProblem.calibrate``.

    After training, this class can be used in conjunction with the fluctuation generation utilities in this package to generate
    realistic turbulence fields based on the learned spectra and eddy lifetimes.
    """

    def __init__(
        self,
        device: str,
        nn_params: NNParameters,
        prob_params: ProblemParameters,
        loss_params: LossParameters,
        phys_params: PhysicalParameters,
        logging_directory: Optional[str] = None,
        output_directory: Union[Path, str] = Path().resolve() / "results",
    ):
        r"""Constructor for ``CalibrationProblem`` class. As depicted in the UML diagram, this requires of 4 dataclasses.

        Parameters
        ----------
        device: str,
            One of the strings ``"cpu", "cuda", "mps"`` indicating the torch device to use
        nn_params : NNParameters
            A ``NNParameters`` (for Neural Network) dataclass instance, which defines values of interest
            eg. size and depth. By default, calls constructor using default values.
        prob_params : ProblemParameters
            A ``ProblemParameters`` dataclass instance, which is used to determine the conditional branching
            and computations required, among other things. By default, calls constructor using default values
        loss_params : LossParameters
            A ``LossParameters`` dataclass instance, which defines the loss function terms and related coefficients.
            By default, calls constructor using the default values.
        phys_params : PhysicalParameters
            A ``PhysicalParameters`` dataclass instance, which defines the physical constants governing the
            problem setting; note that the ``PhysicalParameters`` constructor requires three positional
            arguments.
        output_directory : Union[Path, str], optional
            The directory to write output to; by default ``"./results"``
        """
        self.init_device(device)

        self.nn_params = nn_params
        self.prob_params = prob_params
        self.loss_params = loss_params
        self.phys_params = phys_params

        # stringify the activation functions used; for manual bash only
        self.activfuncstr = str(nn_params.activations)

        self.input_size = nn_params.input_size
        self.hidden_layer_size = nn_params.hidden_layer_sizes

        self.hidden_layer_size = nn_params.hidden_layer_size
        self.init_with_noise = prob_params.init_with_noise
        self.noise_magnitude = prob_params.noise_magnitude

        self.OPS = OnePointSpectra(
            type_eddy_lifetime=self.prob_params.eddy_lifetime,
            physical_params=self.phys_params,
            type_power_spectra=self.prob_params.power_spectra,
            nn_parameters=self.nn_params,
            learn_nu=self.prob_params.learn_nu,
        )

        if self.prob_params.eddy_lifetime == EddyLifetimeType.MANN_APPROX:
            self.OPS.set_scales(
                self.phys_params.L, self.phys_params.Gamma, self.phys_params.sigma
            )

        if self.init_with_noise:
            self.initialize_parameters_with_noise()

        self.log_dimensional_scales()

        self.vdim = 3

        self.epoch_model_sizes = torch.empty((prob_params.nepochs,))

        self.output_directory = output_directory
        self.logging_directory = logging_directory

    # TODO: propagate device setting through this method
    def init_device(self, device: str):
        """Initializes the device (CPU or GPU) on which computation is performed.

        Parameters
        ----------
        device : str
            string following PyTorch conventions -- "cuda" or "cpu"
        """
        self.device = torch.device(device)
        if device == "cuda" and torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # self.OPS.to(self.device)

    # =========================================

    @property
    def parameters(self) -> np.ndarray:
        """Returns all parameters of the One Point Spectra surrogate model as a single vector.

        .. note:: The first 3 parameters of self.parameters() are exactly
            #.  LengthScale

            #.  TimeScale

            #.  Spectrum Amplitude

        Returns
        -------
        np.ndarray
            Single vector containing all model parameters on the CPU, which can be loaded into an object with the same architecture with the
            parameters setter method. This automatically offloads any model parameters that were on the GPU, if any.
        """
        NN_parameters = parameters_to_vector(self.OPS.parameters())

        with torch.no_grad():
            param_vec = (
                NN_parameters.cpu().numpy()
                if NN_parameters.is_cuda
                else NN_parameters.numpy()
            )

        return param_vec

    @parameters.setter
    def parameters(self, param_vec: Union[np.ndarray, torch.tensor]) -> None:
        """Setter method for loading in model parameters from a given vector.

        .. note:: The first 3 parameters of self.parameters() are exactly

            #.  LengthScale

            #.  TimeScale

            #.  Spectrum Amplitude


        Parameters
        ----------
        param_vec : Union[np.ndarray, torch.tensor]
            One-dimensional vector of model parameters.

        Raises
        ------
        ValueError
            "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as network parameters, if using one of TAUNET, CUSTOMMLP."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
        """
        if len(param_vec) < 3:
            raise ValueError(
                "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as network parameters, if using one of TAUNET, CUSTOMMLP."
            )

        if len(param_vec) != len(list(self.parameters)):
            raise ValueError(
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
            )

        if (
            self.OPS.type_EddyLifetime
            in [
                EddyLifetimeType.TAUNET,
                EddyLifetimeType.CUSTOMMLP,
            ]
            and len(param_vec[3:]) != self.num_trainable_params()
        ):
            raise ValueError(
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
            )

        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(
                param_vec, dtype=torch.float64
            )  # TODO: this should also properly load on GPU, issue #28

        vector_to_parameters(param_vec, self.OPS.parameters())

    def log_dimensional_scales(self) -> None:
        """Sets the quantities for non-dimensionalization in log-space.

        .. note:: The first 3 parameters of self.parameters() are exactly

            #.  LengthScale

            #.  TimeScale

            #.  Spectrum Amplitude
        """
        if (
            self.phys_params.L > 0
            and self.phys_params.Gamma > 0
            and self.phys_params.sigma > 0
        ):
            parameters = self.parameters
            parameters[:3] = [
                np.log(self.phys_params.L),
                np.log(self.phys_params.Gamma),
                np.log(self.phys_params.sigma),
            ]

            self.parameters = parameters[: len(self.parameters)]
        else:
            raise ValueError("All dimension scaling constants must be positive.")

    def initialize_parameters_with_noise(self):
        """
        Simple routine to introduce additive white noise to the OPS parameters.
        """
        noise = torch.tensor(
            self.noise_magnitude * torch.randn(*self.parameters.shape),
            dtype=torch.float64,
        )
        vector_to_parameters(noise.abs(), self.OPS.parameters())

        vector_to_parameters(noise, self.OPS.tauNet.parameters())

        vector_to_parameters(noise.abs(), self.OPS.Corrector.parameters())

    def eval(self, k1: torch.Tensor) -> np.ndarray:
        r"""Calls the calibrated model on :math:`k_1`. This can be done after training or after loading trained model parameters from file.

        Parameters
        ----------
        k1 : torch.Tensor
            Tensor of :math:`k_1` data

        Returns
        -------
        np.ndarray
            Evaluation of the model represented in a Numpy array (CPU bound)
        """
        Input = self.format_input(k1)
        with torch.no_grad():
            Output = self.OPS(Input)
        return self.format_output(Output)

    def eval_grad(self, k1: torch.Tensor):
        r"""Evaluates gradient of :math:`k_1` via Autograd

        Parameters
        ----------
        k1 : torch.Tensor
            Tensor of :math::math:`k_1` data

        Returns
        -------
        np.ndarray
            Numpy array of resultant gradient (CPU bound)
        """
        self.OPS.zero_grad()
        Input = self.format_input(k1)
        self.OPS(Input).backward()
        grad = torch.cat([param.grad.view(-1) for param in self.OPS.parameters()])
        return self.format_output(grad)

    def format_input(self, k1: torch.Tensor) -> torch.Tensor:
        r"""Wrapper around clone and cast :math:`k_1` to ``torch.float64``

        Parameters
        ----------
        k1 : torch.Tensor
            Tensor of :math:`k_1`

        Returns
        -------
        torch.Tensor
            Copy of `:math:`k_1` casted to doubles
        """
        formatted_k1 = k1.clone().detach()
        formatted_k1.requires_grad = k1.requires_grad

        return formatted_k1.to(torch.float64)

    def format_output(self, out: torch.Tensor) -> np.ndarray:
        """
        Wrapper around torch's ``out.cpu().numpy()``. Returns a CPU tensor.

        Parameters
        ----------
        out : torch.Tensor
            Tensor to be brought to CPU and converted to an `np.ndarray`

        Returns
        -------
        np.ndarray
            Numpy array of the input tensor
        """
        return out.cpu().numpy()

    # -----------------------------------------

    def calibrate(
        self,
        data: tuple[Iterable[float], torch.Tensor],
        tb_comment: str = "",
        optimizer_class: torch.optim.Optimizer = torch.optim.LBFGS,
    ) -> Dict[str, float]:
        r"""Calibration method, which handles the main training loop and some
        data pre-processing. Currently the only supported optimizer is Torch's ``LBFGS``
        and the learning rate scheduler uses cosine annealing. Parameters for these
        components of the training process are set in ``LossParameters`` and ``ProblemParameters``
        during initialization.

        See the ``.print_calibrated_params()`` method to receive a pretty-printed output of the calibrated physical parameters.

        Parameters
        ----------
        data : tuple[Iterable[float], torch.Tensor]
            Tuple of data points and values, respectively, to be used in calibration.
        tb_comment : str
           Filename comment used by tensorboard; useful for distinguishing between architectures and hyperparameters. Refer to tensorboard documentation for examples of use. By default, the empty string, which results in default tensorboard filenames.
        optimizer_class : torch.optim.Optimizer, optional
           Choice of Torch optimizer, by default torch.optim.LBFGS

        Returns
        -------
        Dict[str, float]
            Physical parameters for the problem, in normal space (not in log-space, as represented internally). Presented as a dictionary in the order ``{L : length scale, Gamma : time scale, sigma : spectrum amplitude}``.

        Raises
        ------
        RuntimeError
            Thrown in the case that the current loss is not finite.
        """

        DataPoints, DataValues = data
        OptimizerClass = optimizer_class
        lr = self.prob_params.learning_rate
        tol = self.prob_params.tol
        nepochs = self.prob_params.nepochs

        self.plot_loss_optim = False

        self.curves = [0, 1, 2, 3]

        self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[:, 0].squeeze()

        self.LossAggregator = LossAggregator(
            params=self.loss_params,
            k1space=self.k1_data_pts,
            zref=self.phys_params.zref,
            tb_log_dir=self.logging_directory,
            tb_comment=tb_comment,
        )

        self.kF_data_vals = torch.cat(
            (
                DataValues[:, 0, 0],
                DataValues[:, 1, 1],
                DataValues[:, 2, 2],
                DataValues[:, 0, 2],
            )
        )

        k1_data_pts, y_data0 = self.k1_data_pts, self.kF_data_vals

        y = self.OPS(k1_data_pts)
        y_data = torch.zeros_like(y)
        y_data[:4, ...] = y_data0.view(4, y_data0.shape[0] // 4)

        ########################################
        # Optimizer and Scheduler Initialization
        ########################################
        if OptimizerClass == torch.optim.LBFGS:
            optimizer = OptimizerClass(
                self.OPS.parameters(),
                lr=lr,
                line_search_fn="strong_wolfe",
                max_iter=self.prob_params.wolfe_iter_count,
                history_size=nepochs,
            )
        else:
            optimizer = OptimizerClass(self.OPS.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)

        self.e_count: int = 0

        if self.OPS.type_EddyLifetime in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
        ]:
            self.gen_theta_NN = lambda: parameters_to_vector(
                self.OPS.tauNet.NN.parameters()
            )
        else:
            self.gen_theta_NN = lambda: 0.0

        theta_NN = self.gen_theta_NN()

        self.loss = self.LossAggregator.eval(
            y[self.curves], y_data[self.curves], theta_NN, 0
        )

        print("=" * 40)

        print(f"Initial loss: {self.loss.item()}")

        print("=" * 40)

        def closure():
            optimizer.zero_grad()
            y = self.OPS(k1_data_pts)

            self.loss = self.LossAggregator.eval(
                y[self.curves], y_data[self.curves], self.gen_theta_NN(), self.e_count
            )

            self.loss.backward()

            self.e_count += 1

            return self.loss

        for _ in tqdm(range(1, nepochs + 1)):
            optimizer.step(closure)
            scheduler.step()

            if not (torch.isfinite(self.loss)):
                raise RuntimeError(
                    "Loss is not a finite value, check initialization and learning hyperparameters."
                )

            if self.loss.item() < tol:
                print(
                    f"Spectra Fitting Concluded with loss below tolerance. Final loss: {self.loss.item()}"
                )
                break

        print("=" * 40)
        print(f"Spectra fitting concluded with final loss: {self.loss.item()}")

        if self.prob_params.learn_nu and hasattr(self.OPS, "tauNet"):
            print(f"Learned nu value: {self.OPS.tauNet.Ra.nu.item()}")

        # physical parameters are stored as natural logarithms internally
        self.calibrated_params = {
            "L       ": np.exp(self.parameters[0]),
            "Γ       ": np.exp(self.parameters[1]),
            "αϵ^{2/3}": np.exp(self.parameters[2]),
        }

        return self.calibrated_params

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def print_calibrated_params(self):
        """Prints out the optimal calibrated parameters ``L``, ``Gamma``, ``sigma``, which are stored in a fitted ``CalibrationProblem`` object under the ``calibrated_params`` dictionary.

        Raises
        ------
        ValueError
            Must call ``.calibrate()`` method to compute a fit to physical parameters first.
        """
        if not hasattr(self, "calibrated_params"):
            raise ValueError(
                "Must call .calibrate() method to compute a fit to physical parameters first."
            )

        OKGREEN = "\033[92m"
        ENDC = "\033[0m"

        print("=" * 40)

        for k, v in self.calibrated_params.items():
            print(f"{OKGREEN}Optimal calibrated {k} : {v:8.4f} {ENDC}")

        print("=" * 40)

        return

    def num_trainable_params(self) -> int:
        """Computes the number of trainable network parameters
            in the underlying model.

            The EddyLifetimeType must be set to one of the following, which involve
            a network surrogate for the eddy lifetime:

                #.  ``TAUNET``

                #.  ``CUSTOMMLP``

        Returns
        -------
        int
            The number of trainable network parameters in the underlying model.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of TAUNET, CUSTOMMLP
        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP."
            )

        return sum(p.numel() for p in self.OPS.tauNet.parameters())

    def eval_trainable_norm(self, ord: Optional[Union[float, str]] = "fro"):
        """Evaluates the magnitude (or other norm) of the trainable parameters in the model.

        .. note::
            The ``EddyLifetimeType`` must be set to one of ``TAUNET`` or ``CUSTOMMLP``, which involve
            a network surrogate for the eddy lifetime.

        Parameters
        ----------
        ord : Optional[Union[float, str]]
            The order of the norm approximation, follows ``torch.norm`` conventions.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of ``TAUNET``, ``CUSTOMMLP``.

        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP."
            )

        return torch.norm(
            torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord
        )

    def eval_trainable_norm(self, ord: Optional[Union[float, str]] = "fro"):
        """Evaluates the magnitude (or other norm) of the
            trainable parameters in the model.

        .. note::
            The ``EddyLifetimeType`` must be set to one of ``TAUNET`` or ``CUSTOMMLP``, which involve
            a network surrogate for the eddy lifetime.

        Parameters
        ----------
        ord : Optional[Union[float, str]]
            The order of the norm approximation, follows ``torch.norm`` conventions.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of ``TAUNET``, ``CUSTOMMLP``.

        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP."
            )

        return torch.norm(
            torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord
        )

    def save_model(self, save_dir: Optional[Union[str, Path]] = None):
        """Saves model with current weights, model configuration, and training histories to file.
        The written filename is of the form ``save_dir/<EddyLifetimeType>_<DataType>.pkl``

        This routine stores

            #.  ``NNParameters``

            #.  ``ProblemParameters``

            #.  ``PhysicalParameters``

            #.  ``LossParameters``

            #.  Optimized Parameters (``self.parameters`` field)

        Parameters
        ----------
        save_dir : Optional[Union[str, Path]], optional
            Directory to save to, by default None; defaults to provided output_dir field for object.

        Raises
        ------
        ValueError
            No output_directory provided during object initialization and no save_dir provided for this method call.
        """

        if save_dir is None and self.output_directory is None:
            raise ValueError(
                "Must provide directory to save output to. Both save_dir and self.output_directory are None"
            )

        if save_dir is None:
            save_dir = self.output_directory

        if isinstance(save_dir, Path):

            if not save_dir.is_dir():
                raise ValueError("Provided save_dir is not actually a directory")

            save_dir = str(save_dir)

        filename = (
            save_dir
            + "/"
            + str(self.prob_params.eddy_lifetime)
            + "_"
            + str(self.prob_params.data_type)
            + ".pkl"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb+") as file:
            pickle.dump(
                [
                    self.nn_params,
                    self.prob_params,
                    self.loss_params,
                    self.phys_params,
                    self.parameters,
                ],
                file,
            )

    def plot(
        self,
        Data: Optional[tuple[Iterable[float], torch.Tensor]] = None,
        model_vals: torch.Tensor = None,
        plot_tau: bool = True,
        save: bool = False,
        save_dir: Optional[Union[Path, str]] = None,
        save_filename: str = "",
    ):
        r"""Plotting method which visualizes the spectra fit as well as the learned eddy lifetime
        function, if ``plot_tau=True``. By default, this operates on the data used in the fitting,
        but an alternative :math:`k_1` domain can be provided and the trained model can be re-evaluated.

        Parameters
        ----------
        Data : tuple[Iterable[float], torch.Tensor], optional
            Tuple of data points and corresponding values, by default ``None``
        model_vals : torch.Tensor, optional
            Evaluation of the OPS on the data, by default None in which case
            ``Data`` must provided (since the function will call OPS on the provided
            ``Data``)
        plot_tau : bool, optional
            Indicates whether to plot the learned eddy lifetime function or not,
            by default ``True``
        save : bool, optional
            Whether to save the resulting figure, by default ``False``
        save_dir : Optional[Union[Path, str]], optional
            Directory to save to, which is created safely if not already present. By default,
            this is the current working directory.
        save_filename : str, optional
            Filename to save the final figure to, by default ``drdmannturb_final_spectra_fit.png``

        Raises
        ------
        ValueError
            Must either provide ``k1space`` or re-use what was used for model calibration;
            thrown in the case neither is specified.
        ValueError
            Must either provide data points or re-use what was used for model calibration;
            thrown in the case neither is specified.
        ValueError
            Thrown in the case that ``save`` is true but neither the ``save_dir`` or ``output_directory``
            are provided.
        """

        clr = ["royalblue", "crimson", "forestgreen", "mediumorchid"]

        if Data is not None:
            DataPoints, DataValues = Data
            k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[:, 0].squeeze()

            kF_data_vals = torch.cat(
                (
                    DataValues[:, 0, 0],
                    DataValues[:, 1, 1],
                    DataValues[:, 2, 2],
                    DataValues[:, 0, 2],
                )
            )
        else:
            if hasattr(self, "k1_data_pts") and self.k1_data_pts is not None:
                k1_data_pts = self.k1_data_pts
            else:
                raise ValueError(
                    "Must either provide k1space or re-use what was used for model calibration, neither is currently specified."
                )

            if hasattr(self, "kF_data_vals") and self.kF_data_vals is not None:
                kF_data_vals = self.kF_data_vals
            else:
                raise ValueError(
                    "Must either provide data points or re-use what was used for model calibration, neither is currently specified."
                )

        kF_model_vals = (
            model_vals
            if model_vals is not None
            else self.OPS(k1_data_pts) / self.phys_params.ustar**2.0
        )

        kF_model_vals = kF_model_vals.cpu().detach()
        kF_data_vals = kF_data_vals.cpu().detach() / self.phys_params.ustar**2

        if plot_tau:
            k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
            k_1 = torch.stack([k_gd, 0 * k_gd, 0 * k_gd], dim=-1)
            k_2 = torch.stack([0 * k_gd, k_gd, 0 * k_gd], dim=-1)
            k_3 = torch.stack([0 * k_gd, 0 * k_gd, k_gd], dim=-1)
            k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3 ** (1 / 2)

        k1_data_pts = k1_data_pts.cpu().detach()

        nrows = 1
        ncols = 2 if plot_tau else 1

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 8})
            self.fig, self.ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                num="Calibration",
                clear=True,
                figsize=[10, 4],
            )
            if not plot_tau:
                self.ax = [self.ax]

            # Subplot 1: One-point spectra
            self.ax[0].set_title("One-point spectra")
            self.lines_SP_model = [None] * (self.vdim + 1)
            self.lines_SP_data = [None] * (self.vdim + 1)
            for i in range(self.vdim):
                (self.lines_SP_model[i],) = self.ax[0].plot(
                    k1_data_pts,
                    kF_model_vals[i].numpy(),
                    "--",
                    color=clr[i],
                    label=r"$F_{0:d}$ model".format(i + 1),
                )

            s = kF_data_vals.shape[0]

            for i in range(self.vdim):
                (self.lines_SP_data[i],) = self.ax[0].plot(
                    k1_data_pts,
                    kF_data_vals.view(4, s // 4)[i].numpy(),
                    "o-",
                    color=clr[i],
                    label=r"$F_{0:d}$ data".format(i + 1),
                    alpha=0.5,
                )
            if 3 in self.curves:
                (self.lines_SP_model[self.vdim],) = self.ax[0].plot(
                    k1_data_pts,
                    -kF_model_vals[self.vdim].numpy(),
                    "--",
                    color=clr[3],
                    label=r"$-F_{13}$ model",
                )
                (self.lines_SP_data[self.vdim],) = self.ax[0].plot(
                    k1_data_pts,
                    -kF_data_vals.view(4, s // 4)[self.vdim].numpy(),
                    "o-",
                    color=clr[3],
                    label=r"$-F_{13}$ data",
                    alpha=0.5,
                )
            self.ax[0].legend()
            self.ax[0].set_xscale("log")
            self.ax[0].set_yscale("log")
            self.ax[0].set_xlabel(r"$k_1 z$")
            self.ax[0].set_ylabel(r"$k_1 F_i /u_*^2$")
            self.ax[0].grid(which="both")

            if plot_tau:
                # Subplot 2: Eddy Lifetime
                self.ax[1].set_title("Eddy lifetime")
                self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
                self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
                self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
                self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()

                self.tau_ref = (
                    self.phys_params.Gamma
                    * MannEddyLifetime(self.phys_params.L * k_gd).cpu().detach().numpy()
                )
                (self.lines_LT_model1,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy() * self.phys_params.L,
                    self.tau_model1,
                    "-",
                    label=r"$\tau_{model}(k_1)$",
                )
                (self.lines_LT_model2,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy() * self.phys_params.L,
                    self.tau_model2,
                    "-",
                    label=r"$\tau_{model}(k_2)$",
                )
                (self.lines_LT_model3,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy() * self.phys_params.L,
                    self.tau_model3,
                    "-",
                    label=r"$\tau_{model}(k_3)$",
                )
                (self.lines_LT_model4,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy() * self.phys_params.L,
                    self.tau_model4,
                    "-",
                    label=r"$\tau_{model}(k,k,k)$",
                )

                (self.lines_LT_ref,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy() * self.phys_params.L,
                    self.tau_ref,
                    "--",
                    label=r"$\tau_{ref}=$Mann",
                )

                self.ax[1].legend()
                self.ax[1].set_xscale("log")
                self.ax[1].set_yscale("log")
                self.ax[1].set_xlabel(r"$k_1 L$")
                self.ax[1].set_ylabel(r"$\tau$")
                self.ax[1].grid(which="both")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        for i in range(self.vdim):
            self.lines_SP_model[i].set_ydata(kF_model_vals[i])
        if 3 in self.curves:
            self.lines_SP_model[self.vdim].set_ydata(-kF_model_vals[self.vdim])

        if plot_tau:
            self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
            self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
            self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
            self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()
            self.lines_LT_model1.set_ydata(self.tau_model1)
            self.lines_LT_model2.set_ydata(self.tau_model2)
            self.lines_LT_model3.set_ydata(self.tau_model3)
            self.lines_LT_model4.set_ydata(self.tau_model4)

        if save:
            if save_dir is not None:
                save_dir = save_dir if type(save_dir) == Path else Path(save_dir)
            elif self.output_directory is not None:
                save_dir = (
                    self.output_directory
                    if type(self.output_directory) == Path
                    else Path(self.output_directory)
                )
            else:
                raise ValueError(
                    "Plot saving is not possible without specifying the save directory or output_directory for the class."
                )

            if save_filename is not None:
                save_path = save_dir / (save_filename + ".png")
            else:
                save_path = save_dir / "drdmannturb_final_spectra_fit.png"

            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            self.fig.savefig(save_path, format="png", dpi=100)

        plt.show()

    def plot_losses(self, run_number: int):
        """A wrapper method around the ``plot_loss_logs`` helper, which plots out the loss
        function terms, multiplied by their associated hyperparameters

        Parameters
        ----------
        run_number : int
            The number of the run in the logging directory to plot out. This is 0-indexed.
        """
        from os import listdir, path

        log_fpath = listdir(self.logging_directory)[run_number]
        full_fpath = path.join(self.logging_directory, log_fpath)

        from drdmannturb import plot_loss_logs

        plot_loss_logs(full_fpath)
