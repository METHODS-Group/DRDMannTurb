"""Provides the CalibrationProblem class, which manages the spectra curve fits."""

from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

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

    Defines the model calibration problem and manages the spectra curve fits.

    Class which manages the spectra fitting and eddy lifetime function learning based on the deep rapid distortion model
    developed in `Keith, Khristenko, Wohlmuth (2021) <https://arxiv.org/pdf/2107.11046.pdf>`_.

    This class manages the operator regression task which characterizes the best fitting candidate
    in a family of nonlocal covariance kernels that are parametrized by a neural network.

    This class can also be used independently of neural networks via the ``EddyLifetimeType`` used for classical spectra
    fitting tasks, for instance, using the ``EddyLifetimeType.MANN`` results in a fit that completely relies on the Mann
    eddy lifetime function. If a neural network model is used, Torch's ``LBFGS`` optimizer is used with cosine annealing
    for learning rate scheduling. Parameters for these components of the training process are set in ``LossParameters``
    and ``ProblemParameters`` during initialization.

    After instantiating ``CalibrationProblem``, wherein the problem and eddy lifetime function substitution
    type are indicated, the user will need to generate the OPS data using ``OnePointSpectraDataGenerator``,
    after which the model can be fit with ``CalibrationProblem.calibrate``.

    After training, this class can be used in conjunction with the fluctuation generation utilities in this package to
    generate realistic turbulence fields based on the learned spectra and eddy lifetimes.
    """  # noqa: D400

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
        r"""Initialize a ``CalibrationProblem`` instance, defining the model calibration and physical setting.

        As depicted in the UML diagram, this requires 4 dataclasses.

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
        logging_directory: Optional[str], optional
            TODO: Add docs for this
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
        self.hidden_layer_size = nn_params.hidden_layer_size
        self.hidden_layer_sizes = nn_params.hidden_layer_sizes

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
            self.OPS.set_scales(self.phys_params.L, self.phys_params.Gamma, self.phys_params.sigma)

        if self.init_with_noise:
            self.initialize_parameters_with_noise()

        self.log_dimensional_scales()

        self.vdim = 3

        self.epoch_model_sizes = torch.empty((prob_params.nepochs,))

        self.output_directory = output_directory
        self.logging_directory = logging_directory

    # TODO: propagate device setting through this method
    def init_device(self, device: str):
        """Initialize the device (CPU or GPU) on which computation is performed.

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
            Single vector containing all model parameters on the CPU, which can be loaded into an object with the same
            architecture with the parameters setter method. This automatically offloads any model parameters that were
            on the GPU, if any.
        """
        NN_parameters = parameters_to_vector(self.OPS.parameters())

        with torch.no_grad():
            param_vec = NN_parameters.cpu().numpy() if NN_parameters.is_cuda else NN_parameters.numpy()

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
            "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as
            network parameters, if using one of TAUNET, CUSTOMMLP."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the
            same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported
            against the currently constructed architecture if this mismatch occurs."
        ValueError
            "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as the
            same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being imported
            against the currently constructed architecture if this mismatch occurs."
        """
        if len(param_vec) < 3:
            raise ValueError(
                "Parameter vector must contain at least 3 dimensionless scale quantities (L, Gamma, sigma) as well as"
                "network parameters, if using one of TAUNET, CUSTOMMLP."
            )

        if len(param_vec) != len(list(self.parameters)):
            raise ValueError(
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as"
                "the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being"
                "imported against the currently constructed architecture if this mismatch occurs."
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
                "Parameter vector must contain values for 3 dimensionless scale quantities (L, Gamma, sigma) as well as"
                "the same number of network parameters, if using one of TAUNET, CUSTOMMLP. Check the architecture being"
                "imported against the currently constructed architecture if this mismatch occurs."
            )

        if not torch.is_tensor(param_vec):
            param_vec = torch.tensor(
                param_vec,  # dtype=torch.float64
            )  # TODO: this should also properly load on GPU, issue #28

        vector_to_parameters(param_vec, self.OPS.parameters())

    def log_dimensional_scales(self) -> None:
        """Set the quantities for non-dimensionalization in log-space.

        .. note:: The first 3 parameters of self.parameters() are exactly

            #.  LengthScale

            #.  TimeScale

            #.  Spectrum Amplitude
        """
        if self.phys_params.L > 0 and self.phys_params.Gamma > 0 and self.phys_params.sigma > 0:
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
        """Introduce additive white noise to the OPS parameters."""
        noise = torch.tensor(
            self.noise_magnitude * torch.randn(*self.parameters.shape),
            # dtype=torch.float64,
        )
        vector_to_parameters(noise.abs(), self.OPS.parameters())

        vector_to_parameters(noise, self.OPS.tauNet.parameters())

        vector_to_parameters(noise.abs(), self.OPS.Corrector.parameters())

    def eval(self, k1: torch.Tensor) -> np.ndarray:
        r"""Evaluate the calibrated model on :math:`k_1`.

        This can be done after training or after loading trained model
        parameters from file.

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
        r"""Evaluate gradient of :math:`k_1` via Autograd.

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
        r"""Cast :math:`k_1` to ``torch.float64``.

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

        return formatted_k1  # .to(torch.float64)

    def format_output(self, out: torch.Tensor) -> np.ndarray:
        """Cast the output to a CPU tensor.

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
    ) -> dict[str, float]:
        r"""Train the model on the provided data.

        Calibration method, which handles the main training loop and some
        data pre-processing. Currently the only supported optimizer is Torch's ``LBFGS``
        and the learning rate scheduler uses cosine annealing. Parameters for these
        components of the training process are set in ``LossParameters`` and ``ProblemParameters``
        during initialization.

        See the ``.print_calibrated_params()`` method to receive a pretty-printed output of the calibrated
        physical parameters.

        Parameters
        ----------
        data : tuple[Iterable[float], torch.Tensor]
            Tuple of data points and values, respectively, to be used in calibration.
        tb_comment : str
           Filename comment used by tensorboard; useful for distinguishing between architectures and hyperparameters.
           Refer to tensorboard documentation for examples of use. By default, the empty string, which results in
           default tensorboard filenames.
        optimizer_class : torch.optim.Optimizer, optional
           Choice of Torch optimizer, by default torch.optim.LBFGS

        Returns
        -------
        Dict[str, float]
            Physical parameters for the problem, in normal space (not in log-space, as represented internally).
            Presented as a dictionary in the order
            ``{L : length scale, Gamma : time scale, sigma : spectrum amplitude}``.

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

        # self.k1_data_pts = torch.tensor(DataPoints, dtype=torch.float64)[:, 0].squeeze()
        self.k1_data_pts = torch.tensor(DataPoints)[:, 0].squeeze()

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
            self.gen_theta_NN = lambda: parameters_to_vector(self.OPS.tauNet.NN.parameters())
        else:
            self.gen_theta_NN = lambda: 0.0

        theta_NN = self.gen_theta_NN()

        self.loss = self.LossAggregator.eval(y[self.curves], y_data[self.curves], theta_NN, 0)

        print("=" * 40)
        print(f"Initial loss: {self.loss.item()}")
        print("=" * 40)

        def closure():
            optimizer.zero_grad()
            y = self.OPS(k1_data_pts)

            self.loss = self.LossAggregator.eval(y[self.curves], y_data[self.curves], self.gen_theta_NN(), self.e_count)

            self.loss.backward()

            self.e_count += 1

            return self.loss

        for _ in tqdm(range(1, nepochs + 1)):
            optimizer.step(closure)
            scheduler.step()

            if not (torch.isfinite(self.loss)):
                raise RuntimeError("Loss is not a finite value, check initialization and learning hyperparameters.")

            if self.loss.item() < tol:
                print(f"Spectra Fitting Concluded with loss below tolerance. Final loss: {self.loss.item()}")
                break

        print("=" * 40)
        print(f"Spectra fitting concluded with final loss: {self.loss.item()}")

        if self.prob_params.learn_nu and hasattr(self.OPS, "tauNet"):
            print(f"Learned nu value: {self.OPS.tauNet.Ra.nu}")

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

    def num_trainable_params(self) -> int:
        """Compute the number of trainable network parameters in the underlying model.

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
            raise ValueError("Not using trainable model for approximation, must be TAUNET, CUSTOMMLP.")

        return sum(p.numel() for p in self.OPS.tauNet.parameters())

    def eval_trainable_norm(self, ord: Optional[Union[float, str]] = "fro"):
        """Evaluate the magnitude (or other norm) of the trainable parameters in the model.

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
            raise ValueError("Not using trainable model for approximation, must be TAUNET, CUSTOMMLP.")

        return torch.norm(torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord)
