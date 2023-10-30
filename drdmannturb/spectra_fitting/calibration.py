"""
This module implements the exposed CalibrationProblem class.
"""

import os
import pickle
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import drdmannturb.loggers as lgg
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


def generic_loss(
    observed: Union[torch.Tensor, float],
    actual: Union[torch.Tensor, float],
    pen: Optional[float] = None,
) -> torch.Tensor:
    """
    Generic loss function implementation

    Parameters
    ----------
    observed : torch.Tensor
        The observed value
    actual : torch.Tensor
        The expected value
    pen : Optional[float], optional
        Penalization term, if any, by default None

    Returns
    -------
    torch.Tensor
        Loss function evaluation
    """

    loss = 0.5 * torch.mean((torch.log(torch.abs(observed / actual))) ** 2)

    if pen is not None:
        loss += pen

    return loss


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
        output_directory: str = "./results",
        logging_level: int = lgg.log.ERROR,
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
        lgg.drdmannturb_log.setLevel(logging_level)

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
        )

        if self.init_with_noise:
            self.initialize_parameters_with_noise()

        self.log_dimensional_scales()

        self.vdim = 3
        self.output_directory = output_directory
        self.fg_coherence = prob_params.fg_coherence
        if (
            self.fg_coherence
        ):  # TODO -- Spectral Coherence needs to be updated with new parameter dataclasses
            self.Coherence = SpectralCoherence(**kwargs)

        self.epoch_model_sizes = torch.empty((prob_params.nepochs,))

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
        optimizer_class: torch.optim.Optimizer = torch.optim.LBFGS,
    ):
        """
        Calibration method, which handles the main training loop and some
        data pre-processing.

        Parameters
        ----------
        data

        model_magnitude_order

        optimizer_class : torch.optim.Optimizer
            User's choice of torch optimizer. By default, LBFGS
        """

        lgg.drdmannturb_log.info("Calibrating MannNet...")

        DataPoints, DataValues = data
        OptimizerClass = optimizer_class
        lr = self.prob_params.learning_rate
        tol = self.prob_params.tol
        nepochs = self.prob_params.nepochs

        self.plot_loss_optim = False
        # self.plot_loss_optim = kwargs.get("plot_loss_wolfe", False)
        # kwargs.get("show", False)

        # TODO -- this should be taken into problem params?
        self.curves = [0, 1, 2, 3]
        # self.curves = kwargs.get("curves", [0, 1, 2, 3])

        alpha_pen1 = self.loss_params.alpha_pen1
        alpha_pen2 = self.loss_params.alpha_pen2
        beta_reg = self.loss_params.beta_reg

        self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[:, 0].squeeze()

        self.LossAggregator = LossAggregator(
            self.loss_params, self.k1_data_pts, self.prob_params.nepochs
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
        # lgg.drdmannturb_log.debug(f"Y_DATA0 shape: {y_data0.shape}")
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

        self.loss_fn = generic_loss

        ##############################
        # Optimizer Set-up
        ##############################
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

        theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())

        self.loss = self.LossAggregator.eval(
            y[self.curves], y_data[self.curves], theta_NN, 0
        )

        print(f"Initial loss: {self.loss.item()}")

        def closure():
            optimizer.zero_grad()
            y = self.OPS(k1_data_pts)
            theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())

            self.loss = self.LossAggregator.eval(
                y[self.curves], y_data[self.curves], theta_NN, self.e_count
            )  # pass epoch

            self.loss.backward()

            self.e_count += 1
            print(self.loss.item())

            return self.loss

        for epoch in range(nepochs):
            optimizer.step(closure)
            scheduler.step()

            if self.loss.item() < tol:
                break

        # torch.nn.Softplus()
        # logk1 = torch.log(self.k1_data_pts).detach()
        # h1 = torch.diff(logk1)
        # h2 = torch.diff(0.5 * (logk1[:-1] + logk1[1:]))
        # torch.diff(0.5 * (self.k1_data_pts[:-1] + self.k1_data_pts[1:]))
        # torch.diff(self.k1_data_pts)

        # def PenTerm(y):
        #     """
        #     TODO: are these embedded functions necessary?
        #     """
        #     logy = torch.log(torch.abs(y))
        #     d2logy = torch.diff(torch.diff(logy, dim=-1) / h1, dim=-1) / h2
        #     # f = torch.nn.GELU()(d2logy) ** 2
        #     f = torch.relu(d2logy).square()
        #     # pen = torch.sum( f * h2 ) / D
        #     pen = torch.mean(f)
        #     return pen

        # def PenTerm1stO(y):
        #     """
        #     TODO: are these embedded functions necessary?
        #     """
        #     logy = torch.log(torch.abs(y))
        #     d1logy = torch.diff(logy, dim=-1) / h1
        #     # d2logy = torch.diff(torch.diff(logy, dim=-1)/h1, dim=-1)/h2
        #     # f = torch.nn.GELU()(d1logy) ** 2
        #     f = torch.relu(d1logy).square()
        #     # pen = torch.sum( f * h2 ) / D
        #     pen = torch.mean(f)
        #     return pen

        # def RegTerm():
        #     """
        #     TODO: are these embedded functions necessary?
        #     """
        #     reg = 0
        #     if self.OPS.type_EddyLifetime == "tauNet":
        #         theta_NN = parameters_to_vector(self.OPS.tauNet.NN.parameters())
        #         reg = theta_NN.square().mean()
        #     return reg

        # def loss_fn(model, target, weights):
        #     """
        #     TODO: are these embedded functions necessary?
        #     """
        #     # y = torch.abs((model-target)).square()

        #     y = torch.log(torch.abs(model / target)).square()

        #     # y = ( (model-target)/(target) ).square()
        #     # y = 0.5*(y[...,:-1]+y[...,1:])
        #     # loss = 0.5*torch.sum( y * h4 )
        #     # loss = torch.sum( y * h1 )

        #     loss = torch.mean(y)
        #     return loss

        # # self.loss_fn = LossFunc()
        # self.loss_fn = loss_fn
        # w = torch.abs(y) / torch.sum(torch.abs(y[:, 0]))
        # self.loss = self.loss_fn(y[self.curves], y_data[self.curves], w[self.curves])

        # # TODO: these should all be torch tensors;
        # # epochs & wolfe iters are known at this point
        # self.loss_history_total = []
        # self.loss_history_epochs = []
        # self.loss_reg = []
        # # TODO: in loss func refactor,
        # # loss_history_total seems to store MSE?
        # # this should rather be stored in loss_mse while
        # # loss_history_total holds the total loss from
        # # all terms
        # self.loss_mse = []
        # self.loss_2ndOpen = []
        # self.loss_1stOpen = []

        # # lgg.drdmannturb_log.simple_optinfo(
        # # f"Initial loss: {self.loss.item()}", tabbed=True
        # # )
        # self.loss_history_total.append(self.loss.item())
        # self.loss_history_epochs.append(self.loss.item())
        # # TODO make sure this doesn't do anything when not using tauNet
        # # self.loss_reg.append(alpha_reg * RegTerm().item())
        # # self.loss_2ndOpen.append(alpha_pen * PenTerm(y).item())
        # # self.loss_1stOpen.append(beta_pen * PenTerm1stO(y).item())

        # # TODO: why is this i needed? just takes all curves?
        # for i in (0,):  # range(len(self.curves),0,-1):

        #     def closure():
        #         optimizer.zero_grad()
        #         y = self.OPS(k1_data_pts)
        #         w = k1_data_pts * y_data / torch.sum(k1_data_pts * y_data)
        #         self.loss = self.loss_fn(
        #             y[self.curves[i:]], y_data[self.curves[i:]], w[self.curves[i:]]
        #         )
        #         if self.fg_coherence:
        #             w1, w2 = (
        #                 1,
        #                 1,
        #             )  ### weights to balance the coherence misfit and others
        #             y_coh = self.Coherence(
        #                 k1_data_pts, Delta_y_data_pts, Delta_z_data_pts
        #             )
        #             loss_coh = self.loss_fn(y_coh, y_coh_data)
        #             self.loss = w1 * self.loss + w2 * loss_coh
        #         self.loss_only = 1.0 * self.loss.item()
        #         self.loss_history_total.append(self.loss_only)
        #         if alpha_pen2:
        #             # adds 2nd order penalty term
        #             pen = alpha_pen2 * PenTerm(y[self.curves[i:]])
        #             # self.loss_2ndOpen.append(pen.item())
        #             self.loss = self.loss + pen
        #             # print('pen = ', pen.item())
        #         if beta_reg:
        #             reg = beta_reg * RegTerm()
        #             # self.loss_reg.append(reg.item())
        #             self.loss = self.loss + reg
        #             # print('reg = ', reg.item())
        #         if alpha_pen1:
        #             # adds 1st order penalty term
        #             pen = alpha_pen1 * PenTerm1stO(y[self.curves[i:]])
        #             # self.loss_1stOpen.append(pen.item())
        #             self.loss += pen
        #         self.loss.backward()
        #         # lgg.drdmannturb_log.simple_optinfo(f"Loss = {self.loss.item()}")

        #         # TODO -- reimplement nu value logging
        #         # if hasattr(self.OPS, 'tauNet'):
        #         #     if hasattr(self.OPS.tauNet.Ra.nu, 'item'):
        #         #         print('-> nu = ', self.OPS.tauNet.Ra.nu.item())
        #         self.kF_model_vals = y.clone().detach()

        #         # self.plot(**kwargs, plt_dynamic=True,
        #         #           model_vals=self.kF_model_vals.cpu().detach().numpy() if torch.is_tensor(self.kF_model_vals) else self.kF_model_vals)
        #         return self.loss

        #     for epoch in range(nepochs):
        #         # lgg.drdmannturb_log.optinfo(f"Epoch {epoch}")
        #         self.epoch_model_sizes[epoch] = self.eval_trainable_norm(
        #             model_magnitude_order
        #         )
        #         optimizer.step(closure)
        #         # TODO: refactor the scheduler things, plateau requires loss
        #         # scheduler.step(self.loss) #if scheduler
        #         scheduler.step()  # self.loss
        #         # self.print_grad()

        #         # lgg.drdmannturb_log.optinfo(self.print_parameters())

        #         self.loss_history_epochs.append(self.loss_only)
        #         if self.loss.item() < tol:
        #             break

        #         if np.isnan(self.loss.item()) or np.isinf(self.loss.item()):
        #             lgg.drdmannturb_log.warning("LOSS IS NAN OR INF")
        #             break

        # lgg.drdmannturb_log.optinfo(
        # f"Calibration terminated with loss = {self.loss.item()} at tol = {tol}",
        # "Calibration",
        # )
        # self.print_parameters()
        # self.plot(plt_dynamic=False)

        # TODO: this should change depending on chosen optimizer;
        # wolfe iterations are only present for LBFGS --
        # split into separate methods
        if self.plot_loss_optim:
            self.plot_loss_wolfe(beta_pen)

        return self.parameters

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def print_parameters(self):
        lgg.drdmannturb_log.warning("[print_parameters] seems deprecated")
        # print(('Optimal NN parameters = [' + ', '.join(['{}'] *
        #       len(self.parameters)) + ']\n').format(*self.parameters))
        pass

    def print_grad(self):
        lgg.drdmannturb_log.warning("[print_grad] seems deprecated")
        # self.grad = torch.cat([param.grad.view(-1)
        #                       for param in self.OPS.parameters()]).detach().numpy()
        # print('grad = ', self.grad)
        pass

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

    def plot_loss_wolfe(self, beta_pen: float = 0.0) -> None:
        """Plots the Wolfe Search loss against the iterations

        Parameters
        ----------
        beta_pen : float, optional
            The loss beta penalty term coefficient; by default, 0.0
        """
        plt.figure()
        plt.plot(self.loss_2ndOpen, label="1st Order Penalty")
        plt.plot(self.loss_reg, label="Regularization")
        plt.plot(self.loss_history_total, label="MSE")

        if beta_pen != 0.0:
            plt.plot(self.loss_1stOpen, label="2nd Order Penalty")

        plt.title("Loss Term Values")

        plt.ylabel("Value")
        plt.xlabel("Wolfe Search Iterations")
        plt.yscale("log")
        plt.legend()
        plt.grid("true")
        plt.show()

    def save_model(self, save_dir: Optional[str] = None):
        """Saves model with current weights, model configuration, and training histories to file.

        File output is of the form save_dir/type_EddyLifetime_data_type.pkl

        Fields that are stored are:
            - NNParameters
            - ProblemParameters
            - PhysicalParameters
            - LossParameters
            - Optimized Parameters (.parameters field)
            - Total Loss History
            - Epoch-wise Loss History

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
                    self.loss_history_total,
                    self.loss_history_epochs,
                ],
                file,
            )

    def plot(self, **kwargs: Dict[str, Any]):
        """
        Handles all plotting
        """
        plt_dynamic = kwargs.get("plt_dynamic", False)

        Data = kwargs.get("Data")
        if Data is not None:
            DataPoints, DataValues = Data
            self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[
                :, 0
            ].squeeze()
            # create a single numpy.ndarray with numpy.array() and then convert to a porch tensor
            # single_data_array=np.array( [DataValues[:, i, i] for i in range(
            # 3)] + [DataValues[:, 0, 2]])
            # self.kF_data_vals = torch.tensor(single_data_array, dtype=torch.float64)
            self.kF_data_vals = torch.cat(
                (
                    DataValues[:, 0, 0],
                    DataValues[:, 1, 1],
                    DataValues[:, 2, 2],
                    DataValues[:, 0, 2],
                )
            )

        k1 = self.k1_data_pts
        torch.stack([0 * k1, k1, 0 * k1], dim=-1)

        plt_tau = kwargs.get("plt_tau", True)
        if plt_tau:
            k_gd = torch.logspace(-3, 3, 50, dtype=torch.float64)
            k_1 = torch.stack([k_gd, 0 * k_gd, 0 * k_gd], dim=-1)
            k_2 = torch.stack([0 * k_gd, k_gd, 0 * k_gd], dim=-1)
            k_3 = torch.stack([0 * k_gd, 0 * k_gd, k_gd], dim=-1)
            k_4 = torch.stack([k_gd, k_gd, k_gd], dim=-1) / 3 ** (1 / 2)
            # k_norm = torch.norm(k, dim=-1)

        self.kF_model_vals = kwargs.get("model_vals", None)
        if self.kF_model_vals is None:
            self.kF_model_vals = self.OPS(k1).cpu().detach().numpy()

        if not hasattr(self, "fig"):
            nrows = 1
            ncols = 2 if plt_tau else 1

            with plt.style.context("bmh"):
                self.fig, self.ax = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    num="Calibration",
                    clear=True,
                    figsize=[10, 5],
                )
                if not plt_tau:
                    self.ax = [self.ax]

                # Subplot 1: One-point spectra
                self.ax[0].set_title("One-point spectra")
                self.lines_SP_model = [None] * (self.vdim + 1)
                self.lines_SP_data = [None] * (self.vdim + 1)
                clr = ["red", "blue", "green", "magenta"]
                for i in range(self.vdim):
                    (self.lines_SP_model[i],) = self.ax[0].plot(
                        k1.cpu().detach().numpy(),
                        self.kF_model_vals[i],
                        color=clr[i],
                        label=r"$F{0:d}$ model".format(i + 1),
                    )  #'o-'

                print(
                    f"k1.size: {k1.size()}   self.kF_data_vals: {self.kF_data_vals.size()}"
                )

                s = self.kF_data_vals.shape[0]

                for i in range(self.vdim):
                    (self.lines_SP_data[i],) = self.ax[0].plot(
                        k1.cpu().detach().numpy(),
                        self.kF_data_vals.view(4, s // 4)[i].cpu().detach().numpy(),
                        "--",
                        color=clr[i],
                        label=r"$F{0:d}$ data".format(i + 1),
                    )
                if 3 in self.curves:
                    (self.lines_SP_model[self.vdim],) = self.ax[0].plot(
                        k1.cpu().detach().numpy(),
                        -self.kF_model_vals[self.vdim],
                        "o-",
                        color=clr[3],
                        label=r"$-F_{13}$ model",
                    )
                    (self.lines_SP_data[self.vdim],) = self.ax[0].plot(
                        k1.cpu().detach().numpy(),
                        -self.kF_data_vals.view(4, s // 4)[self.vdim]
                        .cpu()
                        .detach()
                        .numpy(),
                        "--",
                        color=clr[3],
                        label=r"$-F_{13}$ data",
                    )
                self.ax[0].legend()
                self.ax[0].set_xscale("log")
                self.ax[0].set_yscale("log")
                self.ax[0].set_xlabel(r"$k_1$")
                self.ax[0].set_ylabel(r"$k_1 F_i /u_*^2$")
                self.ax[0].grid(which="both")
                # self.ax[0].set_aspect(1/2)

                if plt_tau:
                    # Subplot 2: Eddy Lifetime
                    self.ax[1].set_title("Eddy lifetime")
                    self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
                    self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
                    self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
                    self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()
                    # self.tau_model1m= self.OPS.EddyLifetime(-k_1).detach().numpy()
                    # self.tau_model2m= self.OPS.EddyLifetime(-k_2).detach().numpy()
                    # self.tau_model3m= self.OPS.EddyLifetime(-k_3).detach().numpy()
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
                    # self.lines_LT_model1m, = self.ax[1].plot(k_gd, self.tau_model1m, '-', label=r'$\tau_{model}(-k_1)$')
                    # self.lines_LT_model2m, = self.ax[1].plot(k_gd, self.tau_model2m, '-', label=r'$\tau_{model}(-k_2)$')
                    # self.lines_LT_model3m, = self.ax[1].plot(k_gd, self.tau_model3m, '-', label=r'$\tau_{model}(-k_3)$')
                    (self.lines_LT_ref,) = self.ax[1].plot(
                        k_gd.cpu().detach().numpy(),
                        self.tau_ref,
                        "--",
                        label=r"$\tau_{ref}=$Mann",
                    )
                    self.ax[1].legend()
                    self.ax[1].set_xscale("log")
                    self.ax[1].set_yscale("log")
                    self.ax[1].set_xlabel(r"$k$")
                    self.ax[1].set_ylabel(r"$\tau$")
                    self.ax[1].grid(which="both")

                    # plt.show()

                # TODO clean up plotting things?
                self.fig.canvas.draw()
                # TODO: comment next out if to save
                self.fig.canvas.flush_events()

            for i in range(self.vdim):
                self.lines_SP_model[i].set_ydata(self.kF_model_vals[i])
            if 3 in self.curves:
                self.lines_SP_model[self.vdim].set_ydata(-self.kF_model_vals[self.vdim])
            # self.ax[0].set_aspect(1)

            if plt_tau:
                self.tau_model1 = self.OPS.EddyLifetime(k_1).cpu().detach().numpy()
                self.tau_model2 = self.OPS.EddyLifetime(k_2).cpu().detach().numpy()
                self.tau_model3 = self.OPS.EddyLifetime(k_3).cpu().detach().numpy()
                self.tau_model4 = self.OPS.EddyLifetime(k_4).cpu().detach().numpy()
                # self.tau_model1m= self.OPS.EddyLifetime(-k_1).detach().numpy()
                # self.tau_model2m= self.OPS.EddyLifetime(-k_2).detach().numpy()
                # self.tau_model3m= self.OPS.EddyLifetime(-k_3).detach().numpy()
                self.lines_LT_model1.set_ydata(self.tau_model1)
                self.lines_LT_model2.set_ydata(self.tau_model2)
                self.lines_LT_model3.set_ydata(self.tau_model3)
                self.lines_LT_model4.set_ydata(self.tau_model4)
                # self.lines_LT_model1m.set_ydata(self.tau_model1m)
                # self.lines_LT_model2m.set_ydata(self.tau_model2m)
                # self.lines_LT_model3m.set_ydata(self.tau_model3m)

                # plt.show()

            if plt_dynamic:
                for ax in self.ax:
                    ax.relim()
                    ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                pass
                # TODO: uncomment next!
                # print("="*30)
                # print("SAVING FINAL SOLUTION RESULTS TO " + f'{self.output_directory+"/" + self.activfuncstr +"final_solution.png"}')

                # self.fig.savefig(self.output_directory+"/" + self.activfuncstr + "final_solution.png", format='png', dpi=100)

                # plt.savefig(self.output_directory.resolve()+'Final_solution.png',format='png',dpi=100)

            # self.fig.savefig(self.output_directory, format='png', dpi=100)
            # self.fig.savefig(self.output_directory.resolve()+"final_solution.png", format='png', dpi=100)
