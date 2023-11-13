"""
This module implements the exposed CalibrationProblem class.
"""

import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from drdmannturb.common import MannEddyLifetime
from drdmannturb.enums import EddyLifetimeType
from drdmannturb.parameters import (
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from drdmannturb.spectra_fitting.one_point_spectra import OnePointSpectra
from drdmannturb.spectra_fitting.spectral_coherence import SpectralCoherence

from .loss_functions import LossAggregator

tqdm = partial(tqdm, position=0, leave=True)


class CalibrationProblem:
    """
    Defines a calibration problem
    """

    def __init__(
        self,
        device: str,
        nn_params: NNParameters,
        prob_params: ProblemParameters,
        loss_params: LossParameters,
        phys_params: PhysicalParameters,
        logging_directory: Optional[str] = None,
        output_directory: str = "./results",
    ):
        """Constructor for CalibrationProblem class. As depicted in the UML diagram, this class consists of 4 dataclasses.

        Parameters
        ----------
        device: str,
            One of the strings "cpu", "cuda", "mps" indicating the torch device to use
        nn_params : NNParameters, optional
            A NNParameters (for Neural Network) dataclass instance, which defines values of interest
            eg. size and depth. By default, calls constructor using default values.
        prob_params : ProblemParameters, optional
            A ProblemParameters dataclass instance, which is used to determine the conditional branching
            and computations required, among other things. By default, calls constructor using default values
        loss_params : LossParameters, optional
            A LossParameters dataclass instance, which defines the loss function terms and related coefficients.
            By default, calls the constructor LossParameters() using the default values.
        phys_params : PhysicalParameters, optional
            A PhysicalParameters dataclass instance, which defines the physical constants governing the
            problem setting; note that the PhysicalParameters constructor requires three positional
            arguments. By default, calls the constructor PhysicalParameters(L=0.59, Gamma=3.9, sigma=3.4).
        output_directory : str, optional
            The directory to write output to; by default "./results"
            TODO: add logging_level docs
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
        self.fg_coherence = prob_params.fg_coherence
        if (
            self.fg_coherence
        ):  # TODO -- Spectral Coherence needs to be updated with new parameter dataclasses
            self.Coherence = SpectralCoherence(**kwargs)

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

        NOTE: The first 3 parameters of self.parameters() are exactly
            - LengthScale
            - TimeScale
            - Magnitude
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

        NOTE: The first 3 parameters of self.parameters() are exactly
            - LengthScale
            - TimeScale
            - Magnitude

        Parameters
        ----------
        param_vec : Union[np.ndarray, torch.tensor]
            One-dimensional vector of model parameters.

        Raises
        ------
        ValueError
            "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
        """
        if len(param_vec) < 3:
            raise ValueError(
                "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET."
            )

        if len(param_vec) != len(list(self.parameters)):
            raise ValueError(
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
            )

        if (
            self.OPS.type_EddyLifetime
            in [
                EddyLifetimeType.TAUNET,
                EddyLifetimeType.CUSTOMMLP,
                EddyLifetimeType.TAURESNET,
            ]
            and len(param_vec[3:]) != self.num_trainable_params()
        ):
            raise ValueError(
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the same number of network parameters, if using one of TAUNET, CUSTOMMLP, or TAURESNET. Check the architecture being imported against the currently constructed architecture if this mismatch occurs."
            )

        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(
                param_vec, dtype=torch.float64
            )  # TODO: this should also properly load on GPU, issue #28

        vector_to_parameters(param_vec, self.OPS.parameters())

    def log_dimensional_scales(self) -> None:
        """Sets the quantities for non-dimensionalization in log-space.

        NOTE: The first 3 parameters of self.parameters() are exactly
            - LengthScale
            - TimeScale
            - Magnitude
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
        TODO -- docstring
        """
        noise = torch.tensor(
            self.noise_magnitude * torch.randn(*self.parameters.shape),
            dtype=torch.float64,
        )
        vector_to_parameters(noise.abs(), self.OPS.parameters())

        vector_to_parameters(noise, self.OPS.tauNet.parameters())

        vector_to_parameters(noise.abs(), self.OPS.Corrector.parameters())

    def eval(self, k1):
        """TODO -- docstring

        Parameters
        ----------
        k1 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        Input = self.format_input(k1)
        with torch.no_grad():
            Output = self.OPS(Input)
        return self.format_output(Output)

    def eval_grad(self, k1):
        """TODO -- docstring

        Parameters
        ----------
        k1 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.OPS.zero_grad()
        Input = self.format_input(k1)
        self.OPS(Input).backward()
        grad = torch.cat([param.grad.view(-1) for param in self.OPS.parameters()])
        return self.format_output(grad)

    def format_input(self, k1: torch.Tensor):
        """Wrapper around clone and cast k1 to torch.float64

        Parameters
        ----------
        k1 : torch.Tensor
            Tensor to be formatted

        Returns
        -------
        torch.Tensor
            Tensor of float64
        """
        formatted_k1 = k1.clone().detach()
        formatted_k1.requires_grad = k1.requires_grad

        return formatted_k1.to(torch.float64)

    def format_output(self, out: torch.Tensor) -> np.ndarray:
        """
        Wrapper around `out.cpu().numpy()`

        Parameters
        ----------
        out : torch.Tensor
            Tensor to be brought to CPU and converted to an np.ndarray

        Returns
        -------
        np.ndarray
            numpy representation of the input tensor
        """
        return out.cpu().numpy()

    # -----------------------------------------

    def calibrate(
        self,
        data: tuple[Any, Any],  # TODO -- properly type this
        model_magnitude_order=1,
        tb_comment: str = "",
        optimizer_class: torch.optim.Optimizer = torch.optim.LBFGS,
    ):
        """Calibration method, which handles the main training loop and some
        data pre-processing.

        Parameters
        ----------
        data : tuple[Any, Any]
            _description_
        tb_comment : str
           Filename comment used by tensorboard; useful for distinguishing between architectures and hyperparameters. Refer to tensorboard documentation for examples of use. By default, the empty string, which results in default tensorboard filenames.
        optimizer_class : torch.optim.Optimizer, optional
           Choice of Torch optimizer, by default torch.optim.LBFGS

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        RuntimeError
            _description_
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
            self.loss_params,
            self.k1_data_pts,
            self.logging_directory,
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

        # The case with the coherence formatting the data

        if self.fg_coherence:  # NOTE: CURRENTLY NOT REACHED B/C ALWAYS FALSE FOR NOW
            DataPoints_coh, DataValues_coh = kwargs.get("Data_Coherence")
            k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts = DataPoints_coh
            k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts = torch.meshgrid(
                k1_data_pts_coh, Delta_y_data_pts, Delta_z_data_pts, indexing="ij"
            )
            y_coh = self.Coherence(k1_data_pts, Delta_y_data_pts, Delta_z_data_pts)
            y_coh_data = torch.zeros_like(y_coh)
            y_coh_data[:] = DataValues_coh

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
            EddyLifetimeType.TAURESNET,
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
            # theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())

            self.loss = self.LossAggregator.eval(
                y[self.curves], y_data[self.curves], self.gen_theta_NN(), self.e_count
            )

            self.loss.backward()

            self.e_count += 1

            return self.loss

        for _ in tqdm(range(1, nepochs + 1)):
            print(self.OPS.tauNet.Ra.nu)
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

        return self.parameters

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def num_trainable_params(self) -> int:
        """Computes the number of trainable network parameters
            in the underlying model.

            The EddyLifetimeType must be set to one of the following, which involve
            a network surrogate for the eddy lifetime:
                - TAUNET
                - CUSTOMMLP
                - TAURESNET

        Returns
        -------
        int
            The number of trainable network parameters in the underlying model.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of TAUNET, CUSTOMMLP, or TAURESNET.
        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
            EddyLifetimeType.TAURESNET,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP, or TAURESNET."
            )

        return sum(p.numel() for p in self.OPS.tauNet.parameters())

    def eval_trainable_norm(self, ord: Optional[Union[float, str]] = "fro"):
        """Evaluates the magnitude (or other norm) of the
            trainable parameters in the model.

            NOTE: The EddyLifetimeType must be set to one of the following, which involve
            a network surrogate for the eddy lifetime:
                - TAUNET
                - CUSTOMMLP
                - TAURESNET

        Parameters
        ----------
        ord : Optional[Union[float, str]]
            The order of the norm approximation, follows ``torch.norm`` conventions.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of TAUNET, CUSTOMMLP, or TAURESNET.

        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
            EddyLifetimeType.TAURESNET,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP, or TAURESNET."
            )

        return torch.norm(
            torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord
        )

    def eval_trainable_norm(self, ord: Optional[Union[float, str]] = "fro"):
        """Evaluates the magnitude (or other norm) of the
            trainable parameters in the model.

            NOTE: The EddyLifetimeType must be set to one of the following, which involve
            a network surrogate for the eddy lifetime:
                - TAUNET
                - CUSTOMMLP
                - TAURESNET

        Parameters
        ----------
        ord : Optional[Union[float, str]]
            The order of the norm approximation, follows ``torch.norm`` conventions.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of TAUNET, CUSTOMMLP, or TAURESNET.

        """
        if self.OPS.type_EddyLifetime not in [
            EddyLifetimeType.TAUNET,
            EddyLifetimeType.CUSTOMMLP,
            EddyLifetimeType.TAURESNET,
        ]:
            raise ValueError(
                "Not using trainable model for approximation, must be TAUNET, CUSTOMMLP, or TAURESNET."
            )

        return torch.norm(
            torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord
        )

    def save_model(self, save_dir: Optional[str] = None):
        """Saves model with current weights, model configuration, and training histories to file.

        File output is of the form save_dir/type_EddyLifetime_data_type.pkl

        Fields that are stored are:
            - NNParameters
            - ProblemParameters
            - PhysicalParameters
            - LossParameters
            - Optimized Parameters (.parameters field)

        Parameters
        ----------
        save_dir : Optional[str], optional
            Directory to save to, by default None; defaults to provided output_dir field for object.

        Raises
        ------
        ValueError
            No output_directory provided during object initialization and no save_dir provided for this method call.
        """

        if save_dir is None and self.output_directory is None:
            raise ValueError("Must provide directory to save output to.")

        if save_dir is None:
            save_dir = self.output_directory

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
        Data=None,
        model_vals=None,
        plot_tau: bool = True,
        save: bool = False,
        save_dir: Optional[Union[Path, str]] = None,
        save_filename: str = "",
    ):
        """Plotting method which visualizes the spectra fit as well as the learned eddy lifetime function, if plot_tau is True. By default, this operates on the data used in the fitting, but an alternative k1 domain can be provided and the trained model can be re-evaluated.

        Parameters
        ----------
        Data : _type_, optional
            _description_, by default None
        model_vals : _type_, optional
            _description_, by default None
        plot_tau : bool, optional
            Whether to plot the learned tau function, by default True
        save : bool, optional
            Whether to save the resulting figure, by default False
        save_dir : Optional[Union[Path, str]], optional
            Directory to save to, which is created safely if not already present. By default, this is the directory in which the code is run.
        save_filename : str, optional
            Filename to save the final figure to, by default drdmannturb_final_spectra_fit.png if no filename is provided here.

        Raises
        ------
        ValueError
            Must either provide k1space or re-use what was used for model calibration, neither is currently specified.
        ValueError
            Must either provide data points or re-use what was used for model calibration, neither is currently specified.
        ValueError
            Plot saving is not possible without specifying the save directory or output_directory for the class.
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

        kF_model_vals = model_vals if model_vals is not None else self.OPS(k1_data_pts)

        kF_model_vals = kF_model_vals.cpu().detach()
        k1_data_pts = k1_data_pts.cpu().detach()
        kF_data_vals = kF_data_vals.cpu().detach()

        if plot_tau:
            k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
            k_1 = torch.stack([k_gd, 0 * k_gd, 0 * k_gd], dim=-1)
            k_2 = torch.stack([0 * k_gd, k_gd, 0 * k_gd], dim=-1)
            k_3 = torch.stack([0 * k_gd, 0 * k_gd, k_gd], dim=-1)
            k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3 ** (1 / 2)

        nrows = 1
        ncols = 2 if plot_tau else 1

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 8})
            self.fig, self.ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                num="Calibration",
                clear=True,
                figsize=[10, 5],
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
                )
            self.ax[0].legend()
            self.ax[0].set_xscale("log")
            self.ax[0].set_yscale("log")
            self.ax[0].set_xlabel(r"$k_1$")
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
                    3.9 * MannEddyLifetime(0.59 * k_gd).cpu().detach().numpy()
                )
                (self.lines_LT_model1,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(),
                    self.tau_model1,
                    "-",
                    label=r"$\tau_{model}(k_1)$",
                )
                (self.lines_LT_model2,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(),
                    self.tau_model2,
                    "-",
                    label=r"$\tau_{model}(k_2)$",
                )
                (self.lines_LT_model3,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(),
                    self.tau_model3,
                    "-",
                    label=r"$\tau_{model}(k_3)$",
                )
                (self.lines_LT_model4,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(),
                    self.tau_model4,
                    "-",
                    label=r"$\tau_{model}(k,k,k)$",
                )

                (self.lines_LT_ref,) = self.ax[1].plot(
                    k_gd.cpu().detach().numpy(),
                    self.tau_ref,
                    "--",
                    label=r"$\tau_{ref}=$Mann",
                )
                self.ax[1].legend()
                self.ax[1].set_xscale("log")
                self.ax[1].set_yscale("log")
                self.ax[1].set_xlabel(r"$f$")
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
