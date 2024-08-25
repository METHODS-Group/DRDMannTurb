# DRDMannTurb

![](https://github.com/METHODS-Group/DRDMannTurb/assets/74629347/604fcde9-41e1-4671-8c10-b1493cadfa88)

DRDMannTurb (short for *Deep Rapid Distortion Theory Mann Turbulence model*) is a data-driven framework
for synthetic turbulence generation in Python.
The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed by Keith et al. in [this 2021 publication](https://arxiv.org/pdf/2107.11046.pdf). 

## Installation 

Pre-compiled wheels for the package are available via ``pip install drdmannturb``. 
See our [development environment instructions](#development-version-installation-instructions)
for instructions on installing development versions.

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

DRDMannTurb source is provided as a locally ``pip``-installable package. We recommended that you use Python 3.9 in your local environment.

To install the package from source without development dependencies, 
it is sufficient to use `pip install -e .`

> [!WARNING]
> Due to current incompatibilities between dependencies and Numpy's API
> changes for version 2.0, DRDMannTurb has currently pinned ``numpy<2.0``
> as a temporary fix.

A Conda environment with all development dependencies can be created off of 
`environment.yml` under `conda_env_files`.

Alternatively, also under `conda_env_files`, you can find explicit exports of complete development
environments for Linux 64-bit and Apple ARM64 macOS. It is not mandatory that you
use these environments.

To recreate the development Conda environments with any of these options, run ``conda create --name <ENV_NAME> --file <conda_environment_file>`` from the
project root. After activating `<ENV_NAME>`, run ``pip install -e .`` to load ``drdmannturb`` into the new environment.

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

Our documentation source lives in the ``/docs/`` folder. You should ensure that the dependencies listed in ``./requirements-docs.txt`` are installed.

Running ``make html`` will generate html pages in the ``/docs/build/html`` folder; these can be hosted locally with ``python -m http.server <PORT-NUMBER>``.

## Acknowledgements

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes. BK was supported in part by the U.S. Department of Energy Office of Science, Early Career Research Program under Award Number DE-SC0024335.
