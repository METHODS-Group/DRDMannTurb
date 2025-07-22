"""Provides the CalibrationProblem class, which manages the spectra curve fits."""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from ..enums import EddyLifetimeType
from ..parameters import (
    IntegrationParameters,
    LossParameters,
    NNParameters,
    PhysicalParameters,
    ProblemParameters,
)
from .data_generator import CustomDataLoader
from .one_point_spectra import OnePointSpectra


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

    data: dict[str, torch.Tensor]  # TODO: This typing may change in the future

    def __init__(
        self,
        nn_params: NNParameters,
        prob_params: ProblemParameters,
        loss_params: LossParameters,
        phys_params: PhysicalParameters,
        integration_params: IntegrationParameters,
        data_loader: CustomDataLoader,
        device: str = "cpu",
        logging_directory: str | None = None,
        output_directory: Path | str = Path().resolve() / "results",
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
            The directory to write output to; by default ``"./results"``
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

        self.OPS = OnePointSpectra(
            type_eddy_lifetime=self.prob_params.eddy_lifetime,
            physical_params=self.phys_params,
            nn_parameters=self.nn_params,
            learn_nu=self.prob_params.learn_nu,
            integration_params=integration_params,
            use_learnable_spectrum=self.prob_params.use_learnable_spectrum,
            p_exponent=self.prob_params.p_exponent,
            q_exponent=self.prob_params.q_exponent,
        )

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

    def calibrate(
        self,
        optimizer_class: torch.optim.Optimizer = torch.optim.LBFGS,
    ) -> dict[str, float]:
        r"""Train the model on the provided data."""
        OptimizerClass = optimizer_class
        lr = self.prob_params.learning_rate
        tol = self.prob_params.tol
        max_epochs = self.prob_params.nepochs

        data = self.data_load.format_data()


        # Plot the OPS data to check that things are correct!








        ## TODO: Prep dataframe columns into tensors for training...

        ########################################
        # Optimizer and Scheduler Initialization
        ########################################
        if OptimizerClass == torch.optim.LBFGS:
            optimizer = OptimizerClass(
                self.OPS.parameters(),
                lr=lr,
                line_search_fn="strong_wolfe",
                max_iter=self.prob_params.wolfe_iter_count,
                history_size=max_epochs,
            )
        else:
            optimizer = torch.optim.Adam(
                self.OPS.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-5,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        self.e_count: int = 0

        if self.OPS.type_EddyLifetime == EddyLifetimeType.TAUNET:
            # NOTE: Old code used tauNet.NN.parameters() since the NN wasn't "built-in" to the tauNet
            self.gen_theta_NN = lambda: parameters_to_vector(self.OPS.tauNet.parameters())
        else:
            self.gen_theta_NN = lambda: 0.0

        theta_NN = self.gen_theta_NN()
        # print("\n[DEBUG calibrate] About to calculate initial loss...")
        # print(f"[DEBUG calibrate] self.curves: {self.curves}")

        self.loss = self.LossAggregator.eval(y[self.curves], y_data[self.curves], theta_NN, 0)

        print("=" * 40)
        print(f"Initial loss: {self.loss.item()}")
        print("=" * 40)

        def closure() -> torch.Tensor:
            """Closure function for the optimizer.

            This function is called by the optimizer to compute the loss and perform a step.
            It is not exposed and should not be otherwise called by the user.
            """
            optimizer.zero_grad()
            y = self.OPS(data_k1_arr)

            coherence_data: dict | None = None
            if self.has_coherence_data:
                model_coh_u, model_coh_v, model_coh_w = self._compute_model_coherence()

                coherence_data = {
                    "model_u": model_coh_u,
                    "model_v": model_coh_v,
                    "model_w": model_coh_w,
                    "data_u": self.coherence_data_u_selected,
                    "data_v": self.coherence_data_v_selected,
                    "data_w": self.coherence_data_w_selected,
                }

            self.loss = self.LossAggregator.eval(
                y[self.curves],
                y_data[self.curves],
                self.gen_theta_NN(),
                self.e_count,
                coherence_data=coherence_data,
            )

            self.loss.backward()
            self.e_count += 1

            return self.loss

        for epoch in range(1, max_epochs + 1):
            # Print the current parameters
            epoch_start_time = time.time()
            print(f"Epoch {epoch} of {max_epochs}")
            print(f"Current loss: {self.loss.item()}")

            optimizer.step(closure)
            scheduler.step()

            epoch_end_time = time.time()
            print(f"Epoch {epoch} took {epoch_end_time - epoch_start_time} seconds")

            print(f"\tCurrent L: {self.OPS.LengthScale.item()}")
            print(f"\tCurrent Gamma: {self.OPS.TimeScale.item()}")
            print(f"\tCurrent sigma: {self.OPS.Magnitude.item()}")
            print(f"\tCurrent p: {self.OPS.p_exponent.item()}")
            print(f"\tCurrent q: {self.OPS.q_exponent.item()}")

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
            "L": np.exp(self.parameters[0]),
            "Γ": np.exp(self.parameters[1]),
            "σ": np.exp(self.parameters[2]),
        }

        if self.prob_params.use_learnable_spectrum:
            self.calibrated_params["p"] = self.parameters[3]
            self.calibrated_params["q"] = self.parameters[4]

        return self.calibrated_params

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def num_trainable_params(self) -> int:
        """Compute the number of trainable network parameters in the underlying model.

            The EddyLifetimeType must be set to one of the following, which involve
            a network surrogate for the eddy lifetime:

                #.  ``TAUNET``

        Returns
        -------
        int
            The number of trainable network parameters in the underlying model.

        Raises
        ------
        ValueError
            If the OPS was not initialized to one of TAUNET
        """
        if self.OPS.type_EddyLifetime != EddyLifetimeType.TAUNET:
            raise ValueError("Not using trainable model for approximation, must be TAUNET.")

        return sum(p.numel() for p in self.OPS.tauNet.parameters())

    def eval_trainable_norm(self, ord: float | str | None = "fro"):
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
        if self.OPS.type_EddyLifetime != EddyLifetimeType.TAUNET:
            raise ValueError("Not using trainable model for approximation, must be TAUNET, CUSTOMMLP.")

        return torch.norm(torch.nn.utils.parameters_to_vector(self.OPS.tauNet.parameters()), ord)

    def save_model(self, save_dir: str | Path | None = None):
        """Pickle and write the trained model to a file.

        Saves model with current weights, model configuration, and training histories to file.
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

        filename = save_dir + "/" + str(self.prob_params.eddy_lifetime) + ".pkl"
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
