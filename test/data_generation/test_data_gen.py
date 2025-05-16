"""Test implementation of data generation functions against v1 DRD data and known values."""

import numpy as np


def test_numerical_closeness(Data, generated_data, rtol=1e-5, atol=1e-6):
    """Tests if the JAX generator data is numerically close to v1 DRD data."""
    # Process DRDMannTurb data
    _, tensors_DRD = Data
    uu_DRD = tensors_DRD[:, 0, 0]
    vv_DRD = tensors_DRD[:, 1, 1]
    ww_DRD = tensors_DRD[:, 2, 2]
    uw_DRD = tensors_DRD[:, 0, 2]

    # Process new JAX data
    phi_jax = generated_data["phi"]
    uu_JAX = phi_jax[:, 0, 0]
    vv_JAX = phi_jax[:, 1, 1]
    ww_JAX = phi_jax[:, 2, 2]
    uw_JAX = phi_jax[:, 0, 2]

    if hasattr(uu_DRD, 'numpy'):
        uu_DRD = uu_DRD.numpy().astype(np.float32)
        vv_DRD = vv_DRD.numpy().astype(np.float32)
        ww_DRD = ww_DRD.numpy().astype(np.float32)
        uw_DRD = uw_DRD.numpy().astype(np.float32)
    else:
        uu_DRD = np.asarray(uu_DRD).astype(np.float32)
        vv_DRD = np.asarray(vv_DRD).astype(np.float32)
        ww_DRD = np.asarray(ww_DRD).astype(np.float32)
        uw_DRD = np.asarray(uw_DRD).astype(np.float32)

    uu_JAX = np.array(uu_JAX).astype(np.float32)
    vv_JAX = np.array(vv_JAX).astype(np.float32)
    ww_JAX = np.array(ww_JAX).astype(np.float32)
    uw_JAX = np.array(uw_JAX).astype(np.float32)

    # --- Assertions ---
    # Shape checks
    assert uu_DRD.shape == uu_JAX.shape, "Shapes of uu components do not match"
    assert vv_DRD.shape == vv_JAX.shape, "Shapes of vv components do not match"
    assert ww_DRD.shape == ww_JAX.shape, "Shapes of ww components do not match"
    assert uw_DRD.shape == uw_JAX.shape, "Shapes of uw components do not match"

    # Dtype checks
    assert uu_DRD.dtype == uu_JAX.dtype, "Dtypes of uu components do not match"
    assert vv_DRD.dtype == vv_JAX.dtype, "Dtypes of vv components do not match"
    assert ww_DRD.dtype == ww_JAX.dtype, "Dtypes of ww components do not match"
    assert uw_DRD.dtype == uw_JAX.dtype, "Dtypes of uw components do not match"

    # Print max abs diff for debugging
    print(f"Max abs diff uu: {np.max(np.abs(uu_DRD - uu_JAX))}")
    print(f"Max abs diff vv: {np.max(np.abs(vv_DRD - vv_JAX))}")
    print(f"Max abs diff ww: {np.max(np.abs(ww_DRD - ww_JAX))}")
    print(f"Max abs diff uw: {np.max(np.abs(uw_DRD - uw_JAX))}")

    # Numerical closeness checks
    assert np.allclose(uu_DRD, uu_JAX, rtol=rtol, atol=atol), f"uu components not close (rtol={rtol}, atol={atol})"
    assert np.allclose(vv_DRD, vv_JAX, rtol=rtol, atol=atol), f"vv components not close (rtol={rtol}, atol={atol})"
    assert np.allclose(ww_DRD, ww_JAX, rtol=rtol, atol=atol), f"ww components not close (rtol={rtol}, atol={atol})"
    # NOTE: Comparing uw directly, not -uw
    assert np.allclose(uw_DRD, uw_JAX, rtol=rtol, atol=atol), f"uw components not close (rtol={rtol}, atol={atol})"

    print("Numerical closeness tests passed successfully!")
    return True