import numpy as np
import pytest

from drdmannturb.fluctuation_generation.low_frequency.fluctuation_field_generator import LowFreqGenerator

# --- Test Configuration ---
L_2D_TEST = 10000.0  # meters
SIGMA2_TEST = 1.0  # m^2/s^2
MIN_SIZE_TARGET = 5 * L_2D_TEST  # 50000.0 m
DEFAULT_PSI = np.deg2rad(45.0)
DEFAULT_ZI = 500.0


def create_base_config(l1_factor=6, l2_factor=6, exp1=8, exp2=8):
    """Helper to create a default config, ensuring values >= 5*L_2d"""
    return {
        "sigma2": SIGMA2_TEST,
        "L_2d": L_2D_TEST,
        "psi": DEFAULT_PSI,
        "z_i": DEFAULT_ZI,
        "L1_factor": l1_factor,
        "L2_factor": l2_factor,
        "exp1": exp1,
        "exp2": exp2,
    }


# --- Helper for Manual Calculation (mirroring _calculate_buffer_sizes) ---
def expected_buffer_sizes(user_l1, user_l2, user_n1, user_n2, l_2d):
    """Manually calculates the expected computational grid dimensions."""
    user_dx = user_l1 / user_n1
    user_dy = user_l2 / user_n2
    min_req_size = 5 * l_2d

    comp_d = min(user_dx, user_dy)
    l1_target = max(user_l1, min_req_size)
    l2_target = max(user_l2, min_req_size)

    n1_min_ideal = int(np.ceil(l1_target / comp_d))
    n2_min_ideal = int(np.ceil(l2_target / comp_d))

    comp_n1 = n1_min_ideal + (n1_min_ideal % 2)
    comp_n2 = n2_min_ideal + (n2_min_ideal % 2)
    if comp_n1 == 0 and n1_min_ideal > 0:
        comp_n1 = 2
    if comp_n2 == 0 and n2_min_ideal > 0:
        comp_n2 = 2

    comp_l1 = comp_n1 * comp_d
    comp_l2 = comp_n2 * comp_d

    return comp_l1, comp_l2, comp_n1, comp_n2


# --- Test Cases ---


@pytest.mark.parametrize(
    "l1_factor, l2_factor, exp1, exp2",
    [
        pytest.param(6, 6, 8, 8, id="Case1_LargerThanMin_Isotropic"),  # 60k x 60k, 256x256
        pytest.param(4, 6, 8, 8, id="Case2_L1_TooSmall"),  # 40k x 60k, 256x256
        pytest.param(6, 4, 8, 8, id="Case3_L2_TooSmall"),  # 60k x 40k, 256x256
        pytest.param(3, 4, 8, 8, id="Case4_Both_TooSmall"),  # 30k x 40k, 256x256
        pytest.param(6, 6, 9, 8, id="Case5_LargerThanMin_AnisotropicRes"),  # 60k x 60k, 512x256
        pytest.param(4, 4, 9, 8, id="Case6_BothSmall_AnisotropicRes"),  # 40k x 40k, 512x256
    ],
)
def test_grid_calculation(l1_factor, l2_factor, exp1, exp2):
    """
    Tests that the computational grid (_calculate_buffer_sizes) is calculated
    correctly based on user input and minimum size requirements.
    """
    config = create_base_config(l1_factor, l2_factor, exp1, exp2)
    gen = LowFreqGenerator(config)

    # Manually calculate expected values
    exp_l1, exp_l2, exp_n1, exp_n2 = expected_buffer_sizes(gen.user_L1, gen.user_L2, gen.user_N1, gen.user_N2, gen.L_2d)

    # Assert calculated computational dimensions match expected
    assert np.isclose(gen.comp_L1, exp_l1), f"comp_L1 mismatch: Got {gen.comp_L1}, Expected {exp_l1}"
    assert np.isclose(gen.comp_L2, exp_l2), f"comp_L2 mismatch: Got {gen.comp_L2}, Expected {exp_l2}"
    assert gen.comp_N1 == exp_n1, f"comp_N1 mismatch: Got {gen.comp_N1}, Expected {exp_n1}"
    assert gen.comp_N2 == exp_n2, f"comp_N2 mismatch: Got {gen.comp_N2}, Expected {exp_n2}"

    # Also verify the calculated spacing is correct
    assert np.isclose(gen.comp_dx, exp_l1 / exp_n1)
    assert np.isclose(gen.comp_dy, exp_l2 / exp_n2)
    assert np.isclose(gen.comp_dx, gen.comp_dy)  # Isotropy check


@pytest.mark.parametrize(
    "l1_factor, l2_factor, exp1, exp2",
    [
        pytest.param(6, 6, 8, 8, id="Case1_LargerThanMin_Isotropic"),
        pytest.param(4, 6, 8, 8, id="Case2_L1_TooSmall"),
        pytest.param(6, 4, 8, 8, id="Case3_L2_TooSmall"),
        pytest.param(3, 4, 8, 8, id="Case4_Both_TooSmall"),
        pytest.param(6, 6, 9, 8, id="Case5_LargerThanMin_AnisotropicRes"),
        pytest.param(4, 4, 9, 8, id="Case6_BothSmall_AnisotropicRes1"),
        pytest.param(2, 3, 9, 8, id="Case7_BothSmall_AnisotropiRes2"),
        pytest.param(3, 2, 9, 8, id="Case8_BothSmall_AnisotropicRes3"),
        pytest.param(2, 2, 9, 8, id="Case9_BothSmall_AnisotropicRes4"),
    ],
)
def test_generate_output_shape_and_variance(l1_factor, l2_factor, exp1, exp2, request, num_realizations=30):
    """
    Tests that generate() returns fields with the correct USER shape and
    that the AVERAGE total variance over the FULL COMPUTATIONAL domain
    is close to the target sigma2.
    """
    config = create_base_config(l1_factor, l2_factor, exp1, exp2)
    target_variance = config["sigma2"]

    # Create the generator instance once, potentially catching the warning
    gen = None
    if config["L1_factor"] * L_2D_TEST < MIN_SIZE_TARGET or config["L2_factor"] * L_2D_TEST < MIN_SIZE_TARGET:
        with pytest.warns(UserWarning, match="User requested domain .* is smaller"):
            gen = LowFreqGenerator(config)
    else:
        gen = LowFreqGenerator(config)

    assert gen is not None, "Generator instance was not created"

    realization_variances_full = []  # Store variance of the full computational field

    # Get the test case ID from the request fixture
    test_case_id = request.node.callspec.id

    # Perform multiple realizations
    for i in range(num_realizations):
        print(f"\n  Running realization {i+1}/{num_realizations} for case id={test_case_id}")
        # Generate returns the extracted fields, but stores the full ones
        u1_extracted, u2_extracted = gen.generate()

        # Check shape of the EXTRACTED fields on first realization
        if i == 0:
            assert u1_extracted.shape == (
                gen.user_N1,
                gen.user_N2,
            ), f"u1 extracted shape mismatch: Got {u1_extracted.shape}, Expected {(gen.user_N1, gen.user_N2)}"
            assert u2_extracted.shape == (
                gen.user_N1,
                gen.user_N2,
            ), f"u2 extracted shape mismatch: Got {u2_extracted.shape}, Expected {(gen.user_N1, gen.user_N2)}"
            assert hasattr(gen, "u1") and gen.u1 is u1_extracted, "gen.u1 not set correctly to extracted field"
            assert hasattr(gen, "u2") and gen.u2 is u2_extracted, "gen.u2 not set correctly to extracted field"
            assert hasattr(gen, "u1_full") and gen.u1_full.shape == (
                gen.comp_N1,
                gen.comp_N2,
            ), "gen.u1_full not set correctly or has wrong shape"
            assert hasattr(gen, "u2_full") and gen.u2_full.shape == (
                gen.comp_N1,
                gen.comp_N2,
            ), "gen.u2_full not set correctly or has wrong shape"

        # Calculate and store total variance from the FULL computational field
        assert hasattr(gen, "u1_full"), "gen.u1_full missing after generate()"
        assert hasattr(gen, "u2_full"), "gen.u2_full missing after generate()"
        total_variance_realization = np.var(gen.u1_full) + np.var(gen.u2_full)
        realization_variances_full.append(total_variance_realization)
        # Reduce print frequency for more realizations
        if (i + 1) % 5 == 0 or i == 0 or i == num_realizations - 1:
            print(f"    Variance (full domain) for realization {i+1}: {total_variance_realization:.4f}")

    # Calculate average variance over all realizations
    average_variance_full = np.mean(realization_variances_full)
    print(f"\n  Average variance (full domain) over {num_realizations} realizations: {average_variance_full:.4f}")
    print(f"  Target variance: {target_variance:.4f}")

    # Check Average Variance (keep the tighter tolerance for now)
    rtol = 0.05  # Allow 5% relative difference for the average
    assert np.isclose(
        average_variance_full, target_variance, rtol=rtol
    ), f"Average variance (full) mismatch: Got {average_variance_full:.4f}, Target {target_variance:.4f} (rtol={rtol})"


# You might want to add specific tests for edge cases if needed,
# e.g., very small N values, L1_factor = 5 exactly, etc.
