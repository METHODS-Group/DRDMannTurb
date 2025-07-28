"""Provides the CalibrationProblem class, which manages the spectra curve fits."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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

    data: dict[str, pl.DataFrame | None]

    def __init__(
        self,
        model: stm.SpectralTensorModel,
        data: dict[str, pl.DataFrame | None] | None = None,
        data_loader: CustomDataLoader | None = None,
        loss_params: LossParameters = LossParameters(),
        integration_params: IntegrationParameters = IntegrationParameters(),
        logging_directory: str | None = None,
        output_directory: Path | str = Path().resolve() / "results",
        device: str = "cpu",
    ):
        r"""Initialize a ``CalibrationProblem`` instance, defining the model calibration and physical setting."""
        self.loss_params = loss_params

        if data is None and data_loader is not None:
            self.data = data_loader.format_data()
        elif data is not None and data_loader is None:
            self.data = data
        else:
            raise ValueError("Provide exactly one of data or data_loader")

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

        # OPS data reconstruction
        ops_data = self.data["ops"]
        assert ops_data is not None, "OPS data is required"

        ops_k_domain_tensor = torch.Tensor(ops_data["freq"])

        # Debug: Check ops_k_domain_tensor
        if torch.isnan(ops_k_domain_tensor).any():
            print(f"NaN in ops_k_domain_tensor: min={ops_k_domain_tensor.min().item()}, max={ops_k_domain_tensor.max().item()}, mean={ops_k_domain_tensor.mean().item()}")
            print(f"ops_k_domain_tensor: {ops_k_domain_tensor}")
            print(f"ops_data['freq']: {ops_data['freq']}")

        ops_uu_true_tensor = torch.Tensor(ops_data["uu"])
        ops_vv_true_tensor = torch.Tensor(ops_data["vv"])
        ops_ww_true_tensor = torch.Tensor(ops_data["ww"])

        # Build the OPS_true tensor
        OPS_true = torch.stack([
            ops_uu_true_tensor,
            ops_vv_true_tensor,
            ops_ww_true_tensor,
        ])

        if OPS_true.isnan().any():
            raise ValueError("OPS data contained NaN values; please remove these from the data.")

        # Build the curves list, which is used to index the model evaluation respecting the order
        # of the returned OPS components.
        curves = [0, 1, 2]

        ops_uw_true_tensor = torch.Tensor(ops_data["uw"])
        if not ops_uw_true_tensor.isnan().any():
            print("Adding uw data to OPS_true")
            OPS_true = torch.cat([OPS_true, ops_uw_true_tensor.unsqueeze(0)], dim=0)
            curves.append(3)
        else:
            print("No uw data provided, skipping.")

        ops_vw_true_tensor = torch.Tensor(ops_data["vw"])
        if not ops_vw_true_tensor.isnan().any():
            OPS_true = torch.cat([OPS_true, ops_vw_true_tensor.unsqueeze(0)], dim=0)
            curves.append(4)
        else:
            print("No vw data provided, skipping.")

        ops_uv_true_tensor = torch.Tensor(ops_data["uv"])
        if not ops_uv_true_tensor.isnan().any():
            OPS_true = torch.cat([OPS_true, ops_uv_true_tensor.unsqueeze(0)], dim=0)
            curves.append(5)
        else:
            print("No uv data provided, skipping.")


        # TODO: Handle the coherence data...
        # coh_data = data["coherence"]

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

        #######################################
        # Helper functions
        #######################################
        def gen_theta_NN() -> torch.Tensor:
            """Obtain any neural network parameters."""
            if hasattr(self.OPS.spectral_tensor_model.eddy_lifetime_model, "tauNet"):
                return parameters_to_vector(self.OPS.spectral_tensor_model.eddy_lifetime_model.tauNet.parameters())
            return torch.tensor(0.0)

        def closure() -> torch.Tensor:
            """Closure function for the optimizer.

            This function is called by the optimizer to compute the loss and perform a step.
            It is not exposed and should not be otherwise called by the user.
            """
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

                # Debug: Check model parameters
                for name, param in self.OPS.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN in parameter {name}: {param}")

                # Debug: Check input tensor
                if torch.isnan(ops_k_domain_tensor).any():
                    print(f"NaN in ops_k_domain_tensor in closure: {ops_k_domain_tensor}")

                OPS_model = self.OPS(ops_k_domain_tensor)

                self.loss = self.LossAggregator.eval(
                    OPS_model[curves],
                    OPS_true,
                    gen_theta_NN(),
                    self.e_count,
                )

                self.loss.backward()

            return self.loss

        close_optims = [torch.optim.LBFGS]

        # Set up real-time plotting
        self._setup_realtime_plotting(ops_data["freq"], OPS_true, curves)

        ###########################################################################################
        # Main optimization loop
        ###########################################################################################
        for epoch in range(1, max_epochs + 1):
            # Print the current parameters
            epoch_start_time = time.time()
            print(f"Epoch {epoch} of {max_epochs}")

            if any(isinstance(optimizer, oc) for oc in close_optims):
                optimizer.step(closure)
                self.e_count += 1

            else:
                with torch.autograd.set_detect_anomaly(True):
                    optimizer.zero_grad()
                    OPS_model = self.OPS(ops_k_domain_tensor)
                    self.loss = self.LossAggregator.eval(
                        OPS_model[curves],
                        OPS_true,
                        gen_theta_NN(),
                        self.e_count,
                    )
                    self.loss.backward()

                    optimizer.step()

                    self.e_count += 1

            scheduler.step()

            epoch_end_time = time.time()
            print(f"Epoch {epoch} took {epoch_end_time - epoch_start_time} seconds")
            print(f"\tCurrent loss: {self.loss.item():.3f}")

            print(f"\tCurrent L: {torch.exp(self.OPS.spectral_tensor_model.log_L).item():.3f}")
            print(f"\tCurrent Gamma: {torch.exp(self.OPS.spectral_tensor_model.log_gamma).item():.3f}")
            print(f"\tCurrent sigma: {torch.exp(self.OPS.spectral_tensor_model.log_sigma).item():.3f}\n")

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

            # Plot every N epochs (adjust as needed)
            if epoch % 5 == 0 or epoch == 1:
                self._update_realtime_plot(epoch)
                plt.pause(0.1)  # Small pause to allow plot to update

        ###########################################################################################
        # Post-optimization
        ###########################################################################################

        plt.ioff()  # Turn off interactive mode
        plt.show()

        print("=" * 40)
        print(f"Spectra fitting concluded with final loss: {self.loss.item()}")

        return

    def _plot_fit_progress(self,
        axes,
        ops_k_domain,
        OPS_true,
        curves,
        epoch,
    ) -> None:
        """Plot the fit progress."""
        with torch.no_grad():
            OPS_model_full = self.OPS(torch.tensor(ops_k_domain))
            OPS_model = OPS_model_full[curves]

        for ax in axes:
            ax.clear()

        component_names = ["uu", "vv", "ww", "uw", "vw", "uv"]
        colors = ["royalblue", "crimson", "forestgreen", "mediumorchid", "orange", "purple"]

        for i, (curve_idx, ax) in enumerate(zip(curves, axes)):
            if i < len(curves):
                ax.plot(ops_k_domain, np.abs(OPS_true[i]), 'o', markersize=3,
                        color=colors[curve_idx], alpha=0.6, label="Data")

                ax.plot(ops_k_domain, OPS_model[i].abs().detach().numpy(), '--',
                        color=colors[curve_idx], label="Model")

                ax.set_title(f"{component_names[curve_idx]} (Epoch {epoch})")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(which="both")
                ax.legend()

        for i in range(len(curves), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.draw()

    def _setup_realtime_plotting(self, ops_k_domain, OPS_true, curves):
        """Set up the real-time plotting with the same styling as the final plot."""
        # Store data for plotting
        self._plot_data = {
            'ops_k_domain': ops_k_domain,
            'OPS_true': OPS_true,
            'curves': curves,
            'epoch': 0
        }

        # Use the same styling as the existing plot method
        plt.ion()  # Turn on interactive mode

        with plt.style.context("bmh"):
            plt.rcParams.update({"font.size": 10})

            # Create figure with same layout as existing plot
            self._fig, self._axes = plt.subplots(
                nrows=2,
                ncols=3,
                num="Real-time Spectra Calibration",
                figsize=[15, 8],
                sharex=True,
            )

            self._axes_flat = self._axes.flatten()

            # Use the same colors and labels as existing plot
            self._clr = ["royalblue", "crimson", "forestgreen", "mediumorchid", "orange", "purple"]
            self._spectra_labels = ["11", "22", "33", "12", "23", "13"]
            self._spectra_names = ["uu", "vv", "ww", "uw", "vw", "uv"]

            # Initialize plot lines
            self._lines_model = [None] * 6
            self._lines_data = [None] * 6

            # Set up initial plot structure
            for i in range(6):
                ax = self._axes_flat[i]

                # Set up axes properties
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$k_1$")
                ax.set_ylabel(rf"$k_1 F_{{{self._spectra_labels[i]}}}(k_1)$")
                ax.grid(which="both")

                # Hide unused subplots
                if i >= len(curves):
                    ax.set_visible(False)
                else:
                    prefix = "auto-" if i < 3 else "cross-"
                    ax.set_title(f"{prefix}spectra {self._spectra_names[i]}")

            self._fig.suptitle("Real-time Spectra Calibration")
            self._fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    def _update_realtime_plot(self, epoch):
        """Update the real-time plot with current model predictions."""
        # Get current model prediction
        with torch.no_grad():
            ops_k_domain_tensor = torch.tensor(self._plot_data['ops_k_domain'])
            OPS_model_full = self.OPS(ops_k_domain_tensor)
            OPS_model = OPS_model_full[self._plot_data['curves']]

        # Update each subplot
        for i, curve_idx in enumerate(self._plot_data['curves']):
            ax = self._axes_flat[i]

            # Clear previous lines
            if self._lines_model[i] is not None:
                self._lines_model[i][0].remove()
            if self._lines_data[i] is not None:
                self._lines_data[i][0].remove()

            # Get data and model values
            model_vals = OPS_model[i].cpu().detach().numpy()
            data_vals = self._plot_data['OPS_true'][i]

            # Take absolute values for log plotting
            model_vals_abs = np.abs(model_vals)
            data_vals_abs = np.abs(data_vals)

            # Plot model values (no sign flipping needed since we're using abs)
            self._lines_model[i] = ax.plot(
                self._plot_data['ops_k_domain'],
                model_vals_abs,
                "--",
                color=self._clr[curve_idx],
                label="Model",
            )

            # Plot data
            self._lines_data[i] = ax.plot(
                self._plot_data['ops_k_domain'],
                data_vals_abs,
                "o",
                markersize=3,
                color=self._clr[curve_idx],
                label="Data",
                alpha=0.6,
            )

            # Update title with epoch info
            prefix = "auto-" if curve_idx < 3 else "cross-"
            ax.set_title(f"{prefix}spectra {self._spectra_names[curve_idx]} (Epoch {epoch})")

        # Update the main title with loss info
        self._fig.suptitle(f"Real-time Spectra Calibration - Epoch {epoch}, Loss: {self.loss.item():.6f}")

        # Redraw
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

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
        clr = ["royalblue", "crimson", "forestgreen", "mediumorchid", "orange", "purple"]
        spectra_labels = ["11", "22", "33", "12", "23", "13"]
        spectra_names = ["uu", "vv", "ww", "uw", "vw", "uv"]

        # Get data
        # data = self.data_loader.format_data()

        ops_data = self.data["ops"]
        assert ops_data is not None, "OPS data is required"

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
