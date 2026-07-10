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

    _OPS_COMPONENTS = ("uu", "vv", "ww", "uw", "vw", "uv")

    data: dict[str, pl.DataFrame | None]
    loss: torch.Tensor

    def __init__(
        self,
        model: stm.SpectralTensorModel,
        loss_params: LossParameters = LossParameters(),
        integration_params: IntegrationParameters = IntegrationParameters(),
        data: dict[str, pl.DataFrame | None] | None = None,
        data_loader: CustomDataLoader | None = None,
        fit_coherence: bool = False,
        fit_components: tuple[str, ...] | None = None,
        logging_directory: str | None = None,
        output_directory: Path | str = Path().resolve() / "results",
        device: str = "cpu",
    ):
        r"""Initialize a ``CalibrationProblem`` instance, defining the model calibration and physical setting."""
        self.loss_params = loss_params
        self.fit_coherence = fit_coherence

        if fit_components is not None:
            if not fit_components:
                raise ValueError("fit_components must contain at least one spectral component")
            if len(set(fit_components)) != len(fit_components):
                raise ValueError("fit_components must not contain duplicates")
            invalid_components = set(fit_components).difference(self._OPS_COMPONENTS)
            if invalid_components:
                raise ValueError(
                    f"Unknown spectral components {sorted(invalid_components)}; "
                    f"expected a subset of {self._OPS_COMPONENTS}"
                )
        self.fit_components = fit_components

        if data is None and data_loader is not None:
            self.data = data_loader.format_data()
        elif data is not None and data_loader is None:
            self.data = data
        else:
            raise ValueError("Provide exactly one of data or data_loader")

        self.OPS = OnePointSpectra(
            spectral_tensor_model=model,
            integration_params=integration_params,
            use_coherence=fit_coherence,
        )

        self.output_directory = output_directory
        self.logging_directory = logging_directory

    def calibrate(
        self,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        lr: float = 1.0,
        optimizer_kwargs: dict | None = None,
        max_epochs: int = 100,
        tol: float = 1e-9,
        auto_scale: bool = True,
        plot_during_calibration: bool = False,
    ) -> None:
        r"""Train the model on the provided data."""
        print("Loading data...")

        # OPS data reconstruction
        ops_data = self.data["ops"]
        assert ops_data is not None, "OPS data is required"

        # Store ALL original data for later use
        self.ops_k_domain_original = torch.Tensor(ops_data["freq"])
        self.ops_uu_original = torch.Tensor(ops_data["uu"])
        self.ops_vv_original = torch.Tensor(ops_data["vv"])
        self.ops_ww_original = torch.Tensor(ops_data["ww"])
        self.ops_uw_original = torch.Tensor(ops_data["uw"])
        self.ops_vw_original = torch.Tensor(ops_data["vw"])
        self.ops_uv_original = torch.Tensor(ops_data["uv"])

        # Initialize scaling factors
        self.freq_scale = 1.0
        self.spectral_scale = 1.0

        if auto_scale:
            freq_scale = torch.median(self.ops_k_domain_original)

            if freq_scale == 0:
                freq_scale = 1.0

        component_data = {
            "uu": self.ops_uu_original,
            "vv": self.ops_vv_original,
            "ww": self.ops_ww_original,
            "uw": self.ops_uw_original,
            "vw": self.ops_vw_original,
            "uv": self.ops_uv_original,
        }

        if self.fit_components is None:
            selected_components = ["uu", "vv", "ww"]
            for component in self._OPS_COMPONENTS[3:]:
                if torch.isfinite(component_data[component]).all():
                    selected_components.append(component)
                    print(f"Adding {component} data to the fit.")
                else:
                    print(f"No complete {component} data provided; skipping.")
        else:
            selected_components = list(self.fit_components)

        for component in selected_components:
            if not torch.isfinite(component_data[component]).all():
                raise ValueError(f"Selected component {component!r} contains NaN or infinite values")

        self.fitted_components = tuple(selected_components)
        print(f"Fitting spectral components: {', '.join(self.fitted_components)}")

        if auto_scale:
            # Scale against only the quantities participating in the objective.
            all_spectral_values = torch.cat([component_data[name] for name in selected_components])

            # Filter out NaN values before computing median
            finite_mask = torch.isfinite(all_spectral_values)
            if finite_mask.any():
                spectral_scale = torch.median(torch.abs(all_spectral_values[finite_mask]))
            else:
                spectral_scale = 1.0

            if spectral_scale == 0:
                spectral_scale = 1.0

            self.freq_scale = freq_scale
            self.spectral_scale = spectral_scale

            print(f"Applied auto-scaling: freq_scale={freq_scale:.3e}, spectral_scale={spectral_scale:.3e}")

        # Use scaled data for training
        ops_k_domain_tensor = self.ops_k_domain_original / self.freq_scale
        OPS_true = torch.stack([component_data[name] / self.spectral_scale for name in selected_components])
        curves = [self._OPS_COMPONENTS.index(name) for name in selected_components]

        # Coherence data preparation (optional)
        coherence_inputs: dict[str, torch.Tensor] | None = None
        if self.fit_coherence:
            coh_df = self.data.get("coherence")
            if coh_df is None:
                print("Coherence fitting requested but no coherence data provided; skipping coherence term.")
            else:
                required_cols = {"r", "freq", "coh_u", "coh_v", "coh_w"}
                missing_cols = required_cols.difference(set(coh_df.columns))
                if missing_cols:
                    raise ValueError(f"Missing required coherence columns: {missing_cols}")

                device = ops_k_domain_tensor.device
                dtype = ops_k_domain_tensor.dtype

                # Coherence data is stored in long form: each frequency appears once
                # per separation and each separation once per frequency.  Building
                # the axes from the full columns duplicates both dimensions and makes
                # every per-separation row look inconsistent with the supposed grid.
                freq_unique = coh_df.get_column("freq").unique().sort().to_list()
                sep_unique = coh_df.get_column("r").unique().sort().to_list()

                unscaled_coh_freqs = torch.tensor(freq_unique, device=device, dtype=dtype)
                coherence_freqs = unscaled_coh_freqs / self.freq_scale
                spatial_seps = torch.tensor(sep_unique, device=device, dtype=dtype)

                def _build_coh_matrix(col: str) -> torch.Tensor:
                    mat = torch.empty((len(sep_unique), len(freq_unique)), device=device, dtype=dtype)
                    for i, sep in enumerate(sep_unique):
                        rows = coh_df.filter(pl.col("r") == sep).sort("freq")
                        freq_for_sep = torch.tensor(rows.get_column("freq").to_list(), device=device, dtype=dtype)
                        if freq_for_sep.numel() != unscaled_coh_freqs.numel() or not torch.allclose(
                            freq_for_sep, unscaled_coh_freqs
                        ):
                            raise ValueError("Coherence frequency grid is inconsistent across separations.")

                        data_vals = torch.tensor(rows.get_column(col).to_list(), device=device, dtype=dtype)
                        mat[i] = data_vals
                    return mat

                data_coh_u = _build_coh_matrix("coh_u")
                data_coh_v = _build_coh_matrix("coh_v")
                data_coh_w = _build_coh_matrix("coh_w")

                coherence_inputs = {
                    "freqs": coherence_freqs,
                    "separations": spatial_seps,
                    "data_u": data_coh_u,
                    "data_v": data_coh_v,
                    "data_w": data_coh_w,
                }

        # Initialize the LossAggregator
        # NOTE: This cannot move to the constructor since it depends on the (scaled) frequency domain.
        self.LossAggregator = LossAggregator(
            params=self.loss_params,
            ops_k_domain=ops_k_domain_tensor,
            tb_log_dir=self.logging_directory,
        )

        ########################################
        # Optimizer and Scheduler Initialization
        ########################################
        optimizer = optimizer_class(
            self.OPS.parameters(),
            lr=lr,
            **(optimizer_kwargs or {}),
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

        def gen_coherence_kwargs() -> dict:
            """Assemble the coherence loss inputs from the current model state, if coherence fitting is enabled."""
            if coherence_inputs is None or self.loss_params.gamma_coherence <= 0:
                return {}

            coh_model = self.OPS.spectral_coherence(
                coherence_inputs["freqs"],
                coherence_inputs["separations"],
            )
            return {
                "coherence_data": {
                    "model_u": coh_model[..., 0],
                    "model_v": coh_model[..., 1],
                    "model_w": coh_model[..., 2],
                    "data_u": coherence_inputs["data_u"],
                    "data_v": coherence_inputs["data_v"],
                    "data_w": coherence_inputs["data_w"],
                }
            }

        def closure() -> torch.Tensor:
            """Closure function for the optimizer.

            This function is called by the optimizer to compute the loss and perform a step.
            It is not exposed and should not be otherwise called by the user.
            """
            optimizer.zero_grad()

            OPS_model = self.OPS(ops_k_domain_tensor)
            if torch.isnan(OPS_model).any():
                raise ValueError("NaN in model output; check initialization and learning hyperparameters.")

            self.loss = self.LossAggregator.eval(
                OPS_model[curves],
                OPS_true,
                gen_theta_NN(),
                self.e_count,
                **gen_coherence_kwargs(),
            )

            self.loss.backward()

            return self.loss

        close_optims = [torch.optim.LBFGS]

        # Set up real-time plotting
        if plot_during_calibration:
            self._setup_realtime_plotting(ops_data["freq"], OPS_true, curves)

        ###########################################################################################
        # Main optimization loop
        ###########################################################################################
        for epoch in range(1, max_epochs + 1):
            epoch_start_time = time.time()

            if any(isinstance(optimizer, oc) for oc in close_optims):
                optimizer.step(closure)
                self.e_count += 1

            else:
                closure()
                optimizer.step()
                self.e_count += 1

            scheduler.step()

            model = self.OPS.spectral_tensor_model
            status = (
                f"Epoch {epoch}/{max_epochs} ({time.time() - epoch_start_time:.2f}s): "
                f"loss = {self.loss.item():.4e}, "
                f"L = {torch.exp(model.log_L).item():.3f}, "
                f"Gamma = {torch.exp(model.log_gamma).item():.3f}, "
                f"sigma = {torch.exp(model.log_sigma).item():.3f}"
            )
            if isinstance(model.energy_spectrum_model, stm.Learnable_ESM):
                esm = model.energy_spectrum_model
                p = esm._positive(esm._raw_p)
                q = esm._positive(esm._raw_q)
                status += f", p = {p.item():.3f}, q = {q.item():.3f}"
            print(status)

            if not (torch.isfinite(self.loss)):
                raise RuntimeError("Loss is not a finite value, check initialization and learning hyperparameters.")

            if self.loss.item() < tol:
                print(f"Spectra Fitting Concluded with loss below tolerance. Final loss: {self.loss.item()}")
                break

            # Plot every N epochs (adjust as needed)
            if plot_during_calibration and (epoch % 5 == 0 or epoch == 1):
                self._update_realtime_plot(epoch)
                plt.pause(0.1)  # Small pause to allow plot to update

        ###########################################################################################
        # Post-optimization
        ###########################################################################################

        if plot_during_calibration:
            plt.ioff()  # Turn off interactive mode
            plt.show()

        print("=" * 40)
        print(f"Spectra fitting concluded with final loss: {self.loss.item()}")

        return

    def _plot_fit_progress(
        self,
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
                ax.plot(
                    ops_k_domain,
                    np.abs(OPS_true[i]),
                    "o",
                    markersize=3,
                    color=colors[curve_idx],
                    alpha=0.6,
                    label="Data",
                )

                ax.plot(ops_k_domain, OPS_model[i].abs().detach().numpy(), "--", color=colors[curve_idx], label="Model")

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
        """Set up the real-time plotting with original data."""
        # Store original data for plotting
        self._plot_data = {
            "ops_k_domain": self.ops_k_domain_original.tolist(),  # Original frequency domain
            "OPS_true": torch.stack(
                [
                    self.ops_uu_original,
                    self.ops_vv_original,
                    self.ops_ww_original,
                    self.ops_uw_original,
                    self.ops_vw_original,
                    self.ops_uv_original,
                ]
            ),
            "curves": curves,
            "epoch": 0,
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
                    curve_idx = curves[i]
                    prefix = "auto-" if curve_idx < 3 else "cross-"
                    ax.set_title(f"{prefix}spectra {self._spectra_names[curve_idx]}")

            self._fig.suptitle("Real-time Spectra Calibration")
            self._fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

    def _update_realtime_plot(self, epoch):
        """Update the real-time plot with current model predictions."""
        with torch.no_grad():
            # Scale input frequency for model
            ops_k_domain_tensor = torch.tensor(self._plot_data["ops_k_domain"]) / self.freq_scale
            OPS_model_full = self.OPS(ops_k_domain_tensor)
            OPS_model = OPS_model_full[self._plot_data["curves"]]

            # Unscale the model predictions
            OPS_model_unscaled = self.unscale_prediction(OPS_model)

        # Update plots using original data and unscaled predictions
        for i, curve_idx in enumerate(self._plot_data["curves"]):
            ax = self._axes_flat[i]

            # Clear previous lines
            if self._lines_model[i] is not None:
                self._lines_model[i][0].remove()
            if self._lines_data[i] is not None:
                self._lines_data[i][0].remove()

            # Use original frequency domain and data
            model_vals = OPS_model_unscaled[i].cpu().detach().numpy()
            data_vals = self._plot_data["OPS_true"][curve_idx]  # Original data

            # Take absolute values for log plotting
            model_vals_abs = np.abs(model_vals)
            data_vals_abs = np.abs(data_vals)

            # Convert to numpy if it's a torch tensor
            if torch.is_tensor(data_vals_abs):
                data_vals_abs = data_vals_abs.cpu().detach().numpy()

            # Plot using original frequency domain
            original_freq = self._plot_data["ops_k_domain"]  # Original units

            self._lines_model[i] = ax.plot(
                original_freq,
                model_vals_abs,
                "--",
                color=self._clr[curve_idx],
                label="Model",
            )

            # Only plot data if it's not all NaN
            if not np.all(np.isnan(data_vals_abs)):
                self._lines_data[i] = ax.plot(
                    original_freq,
                    data_vals_abs,
                    "o",
                    markersize=3,
                    color=self._clr[curve_idx],
                    label="Data",
                    alpha=0.6,
                )

        # Update the main title with loss info
        self._fig.suptitle(f"Real-time Spectra Calibration - Epoch {epoch}, Loss: {self.loss.item():.6f}")

        # Redraw
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ------------------------------------------------
    ### Post-treatment and Export
    # ------------------------------------------------

    def save_model(self, save_path: str | Path | None = None) -> Path:
        """Write the trained spectral tensor model to a checkpoint file.

        In addition to the model itself (see ``SpectralTensorModel.save_model``), the
        checkpoint records the ``freq_scale`` and ``spectral_scale`` factors applied
        during calibration when ``auto_scale=True``, which are needed to map the model
        back to the units of the training data (e.g. when constructing an
        ``NNCovariance`` for fluctuation field generation). These are restored by
        ``SpectralTensorModel.load_model`` as the ``calibration_scales`` attribute.

        Parameters
        ----------
        save_path : Optional[Union[str, Path]], optional
            File to save to, by default None, which writes to ``drd_model.pt`` in this
            object's ``output_directory``. A ``.pt`` suffix is appended if no suffix is
            given.

        Returns
        -------
        Path
            The path the checkpoint was written to.
        """
        if save_path is None:
            save_path = Path(self.output_directory) / "drd_model.pt"
        save_path = Path(save_path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".pt")

        checkpoint = self.OPS.spectral_tensor_model._to_checkpoint()
        checkpoint["calibration"] = {
            "freq_scale": float(getattr(self, "freq_scale", 1.0)),
            "spectral_scale": float(getattr(self, "spectral_scale", 1.0)),
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        return save_path

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

    def unscale_prediction(self, scaled_prediction: torch.Tensor) -> torch.Tensor:
        """Convert scaled model prediction back to original units."""
        return scaled_prediction * self.spectral_scale

    def unscale_frequency(self, scaled_freq: torch.Tensor) -> torch.Tensor:
        """Convert scaled frequency back to original units."""
        return scaled_freq * self.freq_scale

    def get_original_data(self) -> dict:
        """Get the original unscaled data."""
        return {
            "freq": self.ops_k_domain_original,
            "uu": self.ops_uu_original,
            "vv": self.ops_vv_original,
            "ww": self.ops_ww_original,
            "uw": self.ops_uw_original,
            "vw": self.ops_vw_original,
            "uv": self.ops_uv_original,
        }
