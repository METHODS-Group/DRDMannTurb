"""Provides the CalibrationProblem class, which manages the spectra curve fits."""

import time
from pathlib import Path

import torch
from torch.nn.utils import parameters_to_vector

from ..parameters import (
    IntegrationParameters,
    LossParameters,
)
from . import spectral_tensor_models as stm
from .data_generator import CustomDataLoader
from .loss_functions import LossAggregator
from .one_point_spectra import OnePointSpectra


class CalibrationProblem:
    r"""Defines the model calibration problem and manages the spectra curve fits."""

    def __init__(
        self,
        data_loader: CustomDataLoader,
        model: stm.SpectralTensorModel,
        loss_params: LossParameters,
        integration_params: IntegrationParameters,
        logging_directory: str | None = None,
        output_directory: Path | str = Path().resolve() / "results",
        device: str = "cpu",
    ):
        r"""Initialize a ``CalibrationProblem`` instance, defining the model calibration and physical setting."""
        self.loss_params = loss_params

        self.data_loader = data_loader

        self.OPS = OnePointSpectra(
            spectral_tensor_model=model,
            integration_params=integration_params,
        )

        self.output_directory = output_directory
        self.logging_directory = logging_directory

    def calibrate(
        self,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        lr: float = 1.0,
        optimizer_kwargs: dict = {},
        max_epochs: int = 100,
        tol: float = 1e-9,
    ) -> None:
        r"""Train the model on the provided data."""
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

        OPS_true = torch.stack(
            [
                ops_uu_true_tensor,
                ops_vv_true_tensor,
                ops_ww_true_tensor,
                ops_uw_true_tensor,
                ops_vw_true_tensor,
                ops_uv_true_tensor,
            ]
        )

        # TODO: Handle the coherence data...
        # coh_data = data["coherence"]
        del data  # TODO: Is this useful, even?

        ## TODO: Review the coherence data... something is weird with the meshgrid flattening idea
        #        but that SHOULD work...

        # Initialize the LossAggregator
        # TODO: Can't this just go into the constructor?
        self.LossAggregator = LossAggregator(
            params=self.loss_params,
            ops_k_domain=ops_k_domain_tensor,
            tb_log_dir=self.logging_directory,
        )

        OPS_model = self.OPS(ops_k_domain_tensor)

        ########################################
        # Optimizer and Scheduler Initialization
        ########################################
        optimizer = optimizer_class(
            self.OPS.parameters(),
            lr=lr,
            **optimizer_kwargs,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        self.e_count: int = 0

        # TODO: This is a nasty block of code...
        def gen_theta_NN() -> torch.Tensor:
            if hasattr(self.OPS.spectral_tensor_model.eddy_lifetime_model, "tauNet"):
                return parameters_to_vector(self.OPS.spectral_tensor_model.eddy_lifetime_model.tauNet.parameters())
            return torch.tensor(0.0)

        self.loss = self.LossAggregator.eval(OPS_model, OPS_true, gen_theta_NN(), 0)

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
                gen_theta_NN(),
                self.e_count,
            )

            self.loss.backward()
            self.e_count += 1

            for n, p in self.OPS.named_parameters():
                if p.grad is not None:
                    print(f"{n}: {p.grad.norm().item():.3f}")

            return self.loss

        close_optims = [torch.optim.LBFGS]

        for epoch in range(1, max_epochs + 1):
            # Print the current parameters
            epoch_start_time = time.time()
            print(f"Epoch {epoch} of {max_epochs}")
            print(f"Current loss: {self.loss.item()}")

            if any(isinstance(optimizer, oc) for oc in close_optims):
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                OPS_model = self.OPS(ops_k_domain_tensor)
                self.loss = self.LossAggregator.eval(OPS_model, OPS_true, gen_theta_NN(), self.e_count)
                optimizer.step()
                self.e_count += 1

            scheduler.step()

            epoch_end_time = time.time()
            print(f"Epoch {epoch} took {epoch_end_time - epoch_start_time} seconds")

            print(f"\tCurrent L: {torch.exp(self.OPS.spectral_tensor_model.log_L).item():.3f}")
            print(f"\tCurrent Gamma: {torch.exp(self.OPS.spectral_tensor_model.log_gamma).item():.3f}")
            print(f"\tCurrent sigma: {torch.exp(self.OPS.spectral_tensor_model.log_sigma).item():.3f}")

            if isinstance(self.OPS.spectral_tensor_model.energy_spectrum_model, stm.Learnable_ESM):
                p = self.OPS.spectral_tensor_model.energy_spectrum_model._positive(
                    self.OPS.spectral_tensor_model.energy_spectrum_model._raw_p
                )
                q = self.OPS.spectral_tensor_model.energy_spectrum_model._positive(
                    self.OPS.spectral_tensor_model.energy_spectrum_model._raw_q
                )
                print(f"\tCurrent p: {p.item():.3f}")
                print(f"\tCurrent q: {q.item():.3f}")

            if not (torch.isfinite(self.loss)):
                raise RuntimeError("Loss is not a finite value, check initialization and learning hyperparameters.")

            if self.loss.item() < tol:
                print(f"Spectra Fitting Concluded with loss below tolerance. Final loss: {self.loss.item()}")
                break

        print("=" * 40)
        print(f"Spectra fitting concluded with final loss: {self.loss.item()}")

        return

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
        return sum(p.numel() for p in self.OPS.parameters())

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
        return torch.norm(torch.nn.utils.parameters_to_vector(self.OPS.parameters()), ord)

    def save_model(self, save_dir: str | Path | None = None):
        """Pickle and write the trained model to a file.

        Parameters
        ----------
        save_dir : Optional[Union[str, Path]], optional
            Directory to save to, by default None; defaults to provided output_dir field for object.
        """
        raise NotImplementedError("Not implemented")

    def plot(
        self,
    ):
        """Plot the model value against the provided data."""
        import matplotlib.pyplot as plt

        clr = ["royalblue", "crimson", "forestgreen", "mediumorchid", "orange", "purple"]
        spectra_labels = ["11", "22", "33", "12", "23", "13"]
        spectra_names = ["uu", "vv", "ww", "uw", "vw", "uv"]

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
        ops_model = self.OPS(ops_k_domain_tensor)
        ops_model = ops_model.cpu().detach().numpy()

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 10})

            fig_spectra, ax_spectra = plt.subplots(
                nrows=2,
                ncols=3,
                num="Spectra Calibration",
                clear=True,
                figsize=[15, 8],
                sharex=True,
            )

            ax_spectra_flat = ax_spectra.flatten()

            lines_SP_model = [None] * 6
            lines_SP_true = [None] * 6

            for i in range(6):
                ax = ax_spectra_flat[i]

                # Flip the sign of the uw cross-spectra
                sign = -1 if i == 3 else 1

                # Plot model values
                lines_SP_model[i] = ax.plot(
                    ops_k_domain,
                    sign * ops_model[i],
                    "--",
                    color=clr[i],
                    label="Model",
                )

                # Plot data
                lines_SP_true[i] = ax.plot(
                    ops_k_domain,
                    ops_true[i],
                    "o",
                    markersize=3,
                    color=clr[i],
                    label="Data",
                    alpha=0.6,
                )

                prefix = "auto-" if i < 3 else "cross-"
                ax.set_title(f"{prefix}spectra {spectra_names[i]}")
                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.set_xlabel(r"$k_1$")
                ax.set_ylabel(rf"$k_1 F_{{{spectra_labels[i]}}}(k_1)$")
                ax.grid(
                    which="both",
                )

            fig_spectra.suptitle("Calibrated model against data")
            fig_spectra.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
            fig_spectra.canvas.draw()
            fig_spectra.canvas.flush_events()

            # TODO: Redo eddy lifetime plot

        # OPS plot first...
        plt.show()
