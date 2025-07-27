"""Test the data generation utilities."""

from pathlib import Path

import polars as pl
import torch

from drdmannturb.spectra_fitting.data_generator import generate_kaimal_spectra

device = "cpu"

fp = Path(__file__).parent

# def test_kaimal_mann():
#     """Test data generation for Kaimal spectra under Mann parameters from the original 90s paper."""
#     zref = 40  # reference height
#     ustar = 1.773  # friction velocity

#     k1 = torch.logspace(-1, 2, 20) / zref

#     Data = generate_kaimal_spectra(data_points=k1, zref=zref, ustar=ustar)

#     kaimal_mann_spectra_new = Data[1].to("cpu").numpy()

#     kaimal_mann_spectra_true = (
#         torch.load(fp / "kaimal_mann_data_raw.pt", map_location=torch.device(device)).to("cpu").numpy()
#     )

#     assert np.allclose(kaimal_mann_spectra_new, kaimal_mann_spectra_true)

def test_kaimal_data():
    """Test the Polars DataFrame generation for the Kaimal spectra."""
    k1_domain = torch.logspace(-1, 2, 20)

    data_dict_double = generate_kaimal_spectra(k1_domain, zref=1, ustar=1, dtype=pl.Float64)
    data_dict_single = generate_kaimal_spectra(k1_domain, zref=1, ustar=1, dtype=pl.Float32)

    assert isinstance(data_dict_double, dict)
    assert "ops" in data_dict_double and "coherence" in data_dict_double

    ops_data = data_dict_double["ops"]

    # Test the structure of the ops data
    assert ops_data is not None and isinstance(ops_data, pl.DataFrame)
    assert ops_data.shape == (20, 7)
    assert ops_data.columns == ["freq", "uu", "vv", "ww", "uw", "vw", "uv"]
    assert ops_data.dtypes == [pl.Float64] * 7

    # Test the values of the ops data
    # TODO: Write something here

    # TODO: We should implement synthetic coherence data here
    coherence_data = data_dict_double["coherence"]
    assert coherence_data is None


def test_vk_data():
    """Test the Polars DataFrame generation for the von Karman spectra."""
    pass


if __name__ == "__main__":
    test_kaimal_data()
    test_vk_data()
