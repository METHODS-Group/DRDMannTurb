"""Fit the DRD model to the cleaned STORM spectra and coherence data.

Use ``--smoke`` for a small, one-epoch compatibility check before launching a
full research fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn

import drdmannturb as drdmt
from drdmannturb.spectra_fitting import spectral_tensor_models as stm

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = HERE / "data_cleaned"
DEFAULT_OUTPUT_DIRECTORY = HERE / "outputs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--ops-points", type=int, default=100, help="Positive quadrature points per OPS axis")
    parser.add_argument(
        "--coherence-points", type=int, default=100, help="Positive quadrature points per coherence axis"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true", help="Run a small one-epoch end-to-end compatibility check")
    parser.add_argument("--plot", action="store_true", help="Open the fitted one-point spectra plot")
    return parser


def _load_data(data_directory: Path, *, smoke: bool) -> dict[str, pl.DataFrame | None]:
    spectra_file = data_directory / "STORM_downsampled_one_point_spectra.csv"
    coherence_file = data_directory / "STORM_FULL_FIDELITY_coherence_data.csv"
    missing = [path for path in (spectra_file, coherence_file) if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing cleaned STORM input data:\n{missing_text}")

    data = drdmt.CustomDataLoader(
        ops_data_file=spectra_file,
        coherence_data_file=coherence_file,
    ).format_data()

    if smoke:
        ops = data["ops"]
        coherence = data["coherence"]
        assert ops is not None and coherence is not None
        selected_frequencies = coherence.get_column("freq").unique().sort().head(8).to_list()
        selected_separations = coherence.get_column("r").unique().sort().head(3).to_list()
        data = {
            "ops": ops.head(12),
            "coherence": coherence.filter(
                pl.col("freq").is_in(selected_frequencies) & pl.col("r").is_in(selected_separations)
            ),
        }

    return data


def build_problem(
    data: dict[str, pl.DataFrame | None],
    *,
    output_directory: Path,
    ops_points: int,
    coherence_points: int,
) -> drdmt.CalibrationProblem:
    """Construct the STORM calibration problem with the current public API."""
    model = stm.RDT_SpectralTensor(
        eddy_lifetime_model=stm.TauNet_ELT(
            taunet=drdmt.TauNet(
                n_layers=3,
                hidden_layer_sizes=[10, 10, 10],
                activations=[nn.ReLU(), nn.ReLU(), nn.ReLU()],
                learn_nu=False,
            )
        ),
        energy_spectrum_model=stm.Learnable_ESM(p_init=5.0, q_init=3.0),
        L_init=100.0,
        gamma_init=5.0,
        sigma_init=1.0,
    )

    return drdmt.CalibrationProblem(
        model=model,
        data=data,
        loss_params=drdmt.LossParameters(
            alpha_pen1=0.0,
            alpha_pen2=0.0,
            beta_reg=1e-4,
            gamma_coherence=1.0,
        ),
        integration_params=drdmt.IntegrationParameters(
            ops_log_min=-3.0,
            ops_log_max=3.0,
            ops_num_points=ops_points,
            coh_log_min=-3.0,
            coh_log_max=3.0,
            coh_num_points=coherence_points,
        ),
        fit_coherence=True,
        # The homogeneous RDT tensor has structurally zero integrated vw and uv
        # spectra. Keep those exact zeros and fit only represented components.
        fit_components=("uu", "vv", "ww", "uw"),
        logging_directory=str(output_directory / "runs"),
        output_directory=output_directory,
        device="cpu",
    )


def main(argv: list[str] | None = None) -> Path:
    """Run the requested STORM fit and return the checkpoint path."""
    args = _parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.ops_points < 1 or args.coherence_points < 1:
        raise ValueError("quadrature point counts must be at least 1")

    torch.manual_seed(args.seed)
    data = _load_data(args.data_directory.resolve(), smoke=args.smoke)
    epochs = 1 if args.smoke else args.epochs
    ops_points = min(args.ops_points, 4) if args.smoke else args.ops_points
    coherence_points = min(args.coherence_points, 4) if args.smoke else args.coherence_points

    problem = build_problem(
        data,
        output_directory=args.output_directory.resolve(),
        ops_points=ops_points,
        coherence_points=coherence_points,
    )
    learning_rate = min(args.learning_rate, 1e-3) if args.smoke else args.learning_rate
    problem.calibrate(
        optimizer_class=torch.optim.Adam,
        lr=learning_rate,
        optimizer_kwargs={"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-5},
        max_epochs=epochs,
        tol=1e-6,
    )

    checkpoint = problem.save_model(args.output_directory.resolve() / "storm_drd_model.pt")
    print(f"Saved STORM model checkpoint to {checkpoint}")
    if args.plot:
        problem.plot()
    return checkpoint


if __name__ == "__main__":
    main()
