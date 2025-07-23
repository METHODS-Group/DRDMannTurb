"""Provides the CalibrationProblem class, which manages the spectra curve fits."""

import time
from pathlib import Path

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
from .loss_functions import LossAggregator
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
        logging_directory: str | None = None,
        output_directory: Path | str = Path().resolve() / "results",
        device: str = "cpu",
    ):
        r"""Initialize a ``CalibrationProblem`` instance, defining the model calibration and physical setting."""
        self.nn_params = nn_params
        self.prob_params = prob_params
        self.loss_params = loss_params
        self.phys_params = phys_params

        self.data_loader = data_loader

        self.OPS = OnePointSpectra(
            type_eddy_lifetime=self.prob_params.eddy_lifetime,
            physical_params=self.phys_params,
            nn_parameters=self.nn_params,
            learn_nu=self.prob_params.learn_nu,
            integration_params=integration_params,
            # The following are only used if the user wants to learn the VK energy spectrum exponents
            use_learnable_spectrum=self.prob_params.use_learnable_spectrum,
            p_exponent=self.prob_params.p_exponent,
            q_exponent=self.prob_params.q_exponent,
        )

        self.output_directory = output_directory
        self.logging_directory = logging_directory

    def calibrate(
        self,
        optimizer_class: torch.optim.Optimizer = torch.optim.LBFGS,
    ) -> dict[str, float]:
        r"""Train the model on the provided data."""
        OptimizerClass = optimizer_class
        lr = self.prob_params.learning_rate
        tol = self.prob_params.tol
        max_epochs = self.prob_params.nepochs

        # Load the data
        print("Loading data...")
        data = self.data_loader.format_data()

        # OPS data reconstruction
        ops_data = data["ops"]

        ops_k_domain_tensor = torch.Tensor(ops_data["freq"])

        ops_uu_true_tensor = torch.Tensor(ops_data["uu"])
        ops_vv_true_tensor = torch.Tensor(ops_data["vv"])
        ops_ww_true_tensor = torch.Tensor(ops_data["ww"])

        # TODO: This does NOT currently handle cases where the user has not provided all the cross-spectra data
        ops_uw_true_tensor = torch.Tensor(ops_data["uw"])
        ops_vw_true_tensor = torch.Tensor(ops_data["vw"])
        ops_uv_true_tensor = torch.Tensor(ops_data["uv"])

        OPS_true = torch.stack([
            ops_uu_true_tensor,
            ops_vv_true_tensor,
            ops_ww_true_tensor,
            ops_uw_true_tensor,
            ops_vw_true_tensor,
            ops_uv_true_tensor,
        ])

        # TODO: Handle the coherence data...
        coh_data = data["coherence"]
        del data # TODO: Is this useful, even?

        ## TODO: Review the coherence data... something is weird with the meshgrid flattening idea
        #        but that SHOULD work...

        # Initialize the LossAggregator
        # TODO: Can't this just go into the constructor?
        self.LossAggregator = LossAggregator(
            params = self.loss_params,
            ops_k_domain = ops_k_domain_tensor,
            tb_log_dir = self.logging_directory,
        )

        OPS_model = self.OPS(ops_k_domain_tensor)

        ########################################
        # Optimizer and Scheduler Initialization
        ########################################
        # TODO: Clean up how the optimizer is taken in...
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

        self.loss = self.LossAggregator.eval(OPS_model, OPS_true, theta_NN, 0)

        print("=" * 40)
        print(f"Initial loss: {self.loss.item()}")
        print("=" * 40)

        def closure() -> torch.Tensor:
            """Closure function for the optimizer.

            This function is called by the optimizer to compute the loss and perform a step.
            It is not exposed and should not be otherwise called by the user.
            """
            optimizer.zero_grad()
            OPS_model = self.OPS(ops_k_domain_tensor)

            self.loss = self.LossAggregator.eval(
                OPS_model,
                OPS_true,
                self.gen_theta_NN(),
                self.e_count,
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

        # TODO: Need to add some method to grab the length scale, time scale, and magnitude from OPS
        # self.calibrated_params = {
        #     "L": np.exp(self.parameters[0]),
        #     "Γ": np.exp(self.parameters[1]),
        #     "σ": np.exp(self.parameters[2]),
        # }

        # if self.prob_params.use_learnable_spectrum:
            # self.calibrated_params["p"] = self.parameters[3]
            # self.calibrated_params["q"] = self.parameters[4]

        return
        # return self.calibrated_params

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
        raise NotImplementedError("Not implemented")

    def plot(self,):
        """Plot the model value against the provided data."""
        import matplotlib.pyplot as plt

        # Get data
        data = self.data_loader.format_data()

        ops_data = data["ops"]
        ops_k_domain = ops_data["freq"]
        ops_k_domain_tensor = torch.Tensor(ops_data["freq"])

        ops_true = [
            ops_data["uu"],
            ops_data["vv"],
            ops_data["ww"],
            ops_data["uw"],
            ops_data["vw"],
            ops_data["uv"],
        ]

        # Get model OPS
        ops_model = ops_k_domain_tensor * self.OPS(ops_k_domain_tensor)
        ops_model = ops_model.cpu().detach().numpy()

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 10})

            fig_spectra, ax_spectra = plt.subplots(
                nrows = 2,
                ncols = 3,
                num="Spectra Calibration",
                clear=True,
                figsize=[15,8],
                sharex=True,
            )

            ax_spectra_flat = ax_spectra.flatten()

            lines_SP_model = [None] * 6
            lines_SP_true = [None] * 6

            for i in range(6):
                ax = ax_spectra_flat[i]

                lines_SP_model[i] = ax.plot(
                    ops_k_domain,
                    ops_model[i],
                    label="Model",
                )

                lines_SP_true[i] = ax.plot(
                    ops_k_domain,
                    ops_true[i],
                    label="Data",
                )

                title = rf"OPS component {i+1}"
                ax.set_title(title)
                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xlabel(r"$k_1$")
                ax.set_ylabel(r"$k_1 F_{ij}(k_1)$")
                ax.grid(which="both",)

            # TODO: Redo eddy lifetime plot





        # OPS plot first...
        plt.show()
