# STORM fitting workflow

`dry_run.py` is the canonical command-line entry point for fitting the current
DRDMannTurb model to the cleaned STORM one-point spectra and spatial coherence
data. The older `debug.py` and `script_fit_coh.py` names remain as thin wrappers
so existing notes and shell commands continue to work.

From the repository root, first run the compatibility check:

```bash
uv run --group storm python STORM_fitting/dry_run.py --smoke
```

The smoke run uses 12 one-point spectra samples, an 8-by-3 coherence grid,
reduced quadrature, and one optimizer epoch. It exercises data loading, both
spectra integrators, coherence loss, backpropagation, and checkpoint writing.

The current homogeneous RDT tensor fits `uu`, `vv`, `ww`, and `uw`. Its
integrated `vw` and `uv` components are structurally zero, so those measured
columns are retained in the input data but explicitly excluded from this fit.
No epsilon floor or clamping is applied to spectra or targets.

Launch the research-sized fit with:

```bash
uv run --group storm python STORM_fitting/dry_run.py \
  --epochs 20 \
  --ops-points 100 \
  --coherence-points 100
```

Outputs are written below `STORM_fitting/outputs/` by default. Use `--plot` to
open the one-point spectra comparison after fitting. Run `--help` for all path,
resolution, learning-rate, and seed options.

The cleaned CSV files are research data and are intentionally excluded by the
repository-wide `*.csv` ignore rule. A checkout intended to run this workflow
must provide:

- `data_cleaned/STORM_downsampled_one_point_spectra.csv`
- `data_cleaned/STORM_FULL_FIDELITY_coherence_data.csv`
