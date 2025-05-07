# DRDMannTurb

[![DOI](https://zenodo.org/badge/649495955.svg)](https://doi.org/10.5281/zenodo.13922330)

[![DOI](https://camo.githubusercontent.com/8297af3b00072c1c2b0c4357a0ba3c8f3ac2517f4eb737eb020f3fa35fbdd13e/68747470733a2f2f6a6f73732e7468656f6a2e6f72672f7061706572732f31302e32313130352f6a6f73732e30363833382f7374617475732e737667)](https://doi.org/10.21105/joss.06838)

![](https://github.com/METHODS-Group/DRDMannTurb/assets/74629347/604fcde9-41e1-4671-8c10-b1493cadfa88)

DRDMannTurb (short for *Deep Rapid Distortion Theory Mann Turbulence model*) is a data-driven framework
for synthetic turbulence generation in Python.
The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed by Keith et al. in [this 2021 publication](https://arxiv.org/pdf/2107.11046.pdf).

## Installation

Pre-compiled wheels for the package are available via ``pip install drdmannturb``.
See our [development environment instructions](#development-version-installation-instructions)
for instructions on installing development versions.

> [!NOTE]
> `DRDMannTurb` requires the `FFTW` library to be installed on your system for the `pyFFTW` dependency. Please install it using your system's package manager (e.g., `brew install fftw` on macOS, `sudo apt-get install libfftw3-dev` on Debian/Ubuntu) before installing `DRDMannTurb`.

## Basic Usage

See the ``/examples/`` folder for baselines from the paper and for examples of the many functionalities of the package. These examples are rendered in a more readable
format on our documentation [here](https://methods-group.github.io/DRDMannTurb/examples.html) also.

DRDMannTurb consists of two primary submodules ``spectra_fitting`` and ``fluctuation_generation``
which are respectively focused on fitting a Deep Rapid Distortion (DRD) model and
on generating synthetic turbulence "boxes" with a fit DRD model.

## Questions?

If you have any questions, the best way to receive help is by creating a thread in our Discussions or by contacting the authors (Alexey Izmailov, Matthew Meeker) by email directly. If your question pertains to a problem with the package, please open an Issue so that it can addressed.

## Citation

If you use this software, please cite it as below.

```
@software{Izmailov_DRDMannTurb_2023,
author = {Izmailov, Alexey and Meeker, Matthew and Deskos, Georgios and Keith, Brendan},
month = mar,
title = {{DRDMannTurb}},
url= {https://github.com/METHODS-Group/DRDMannTurb},
version = {1.0.2},
year = {2024}
}
```

## Contributing

We always welcome new contributors! The best way to contribute to DRDMannTurb is through opening an issue, making a feature request, or creating a pull request directly.

See also the below instructions for installing DRDMannTurb for development purposes.

### Development Version Installation Instructions

To set up a development environment, we recommend using [`uv`](https://github.com/astral-sh/uv), a fast Python package installer and resolver.

1.  **Install `uv`**: Follow the official [`uv` installation instructions](https://github.com/astral-sh/uv#installation).
2.  **Install System Dependencies**: Ensure the `FFTW` library is installed (see [Installation](#installation) section above).
3.  **Install Python**: Make sure you have Python >= 3.9.16 installed and available. We recommend using a Python version manager like [`pyenv`](https://github.com/pyenv/pyenv).
4.  **Clone the Repository**: `git clone https://github.com/METHODS-Group/DRDMannTurb.git && cd DRDMannTurb`
5.  **Create and Activate Virtual Environment**:
    ```bash
    # Create a virtual environment using your installed Python
    uv venv
    # Alternatively, you can explicitly set the Python version with the following
    uv venv --python 3.9.16
    # Activate the environment (syntax may vary slightly for different shells)
    source .venv/bin/activate
    ```
6.  **Install in Editable Mode with Dependencies**: Install the package in editable mode along with development and documentation dependencies. *Make sure to quote the argument containing brackets.*
    ```bash
    # Install core, docs, and dev dependencies
    uv pip install -e '.[docs,dev]'
    ```

We also ask that you install our
``pre-commit`` configuration by running ``pre-commit install`` in the root directory
of this repository. If you are unfamiliar with ``pre-commit``,
the documentation can be found [here](https://pre-commit.com/).

### Running Tests Locally

DRDMannTurb's test suite is built with [Pytest](https://docs.pytest.org/en/stable/). Running the tests locally can be done by running `pytest`
from the project root.

Tests decorated with `slow` can be run with the `--runslow` flag; they are otherwise skipped. Note that several of these tests require (at least
``partially'') training a DRD model, and so the suite may take several minutes to complete.
Note also that certain components of the test suite require CUDA; these are also
skipped if a CUDA device is not available.

### Local Documentation Building Instructions

Our documentation source lives in the ``/docs/`` folder.
Ensure you have installed the necessary dependencies by including the `docs` extra during installation (`uv pip install -e '.[docs,dev]'`).

Running ``make html`` from the `docs` directory will generate html pages in the ``/docs/build/html`` folder; these can be hosted locally with ``python -m http.server <PORT-NUMBER>``.

## Acknowledgements

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes. BK was supported in part by the U.S. Department of Energy Office of Science, Early Career Research Program under Award Number DE-SC0024335.
