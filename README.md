# Deep Rapid Distortion theory Mann Turbulence model

The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed by Keith et al in [this 2021 publication](https://arxiv.org/pdf/2107.11046.pdf). 

## Basic Usage


See the ``/examples/`` folder for baselines from the paper and for examples of the many functionalities of the package.

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``drdmannturb`` into your environment. 

We also suggest installing the local ``pre-commit`` configuration by running ``pre-commit install`` in the root directory of this repository. 

## Local Documentation Building Instructions 

Docs are in the ``/docs/`` folder. Make sure the dependencies listed in ``./requirements-docs.txt`` are installed.

Run ``make html`` to generate html pages in the ``/docs/build/html`` folder, which can be hosted locally with ``python -m http.server <PORT-NUMBER>``. 

## Attribution

