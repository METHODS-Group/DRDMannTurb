# DRDMannTurb

`DRDMannTurb` (Deep Rapid Distortion Mann Turbulence) is a GPU-accelerated, data-driven Python framework for the auomatic fitting
of spectra data and generation of synthetic turbulent wind fields. The package is intended to be used by wind engineers in requiring
rapid simulation of realistic wind turbulence.

The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed in [Keith, Khristenko, Wohlmuth (2021)](https://arxiv.org/pdf/2107.11046.pdf). 

To learn more and see some examples, visit the [documentation]() or `/examples`.


## Installation


## Features




## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. We recommend that you use Python 3.9 in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``drdmannturb`` into your environment.

### Pre-commit hooks

We use `pre-commit`. See 

## Local Documentation Building Instructions 

Docs are in the ``/docs/`` folder. Make sure the dependencies listed in ``./requirements-docs.txt`` are installed.

Run ``make html`` to generate html pages in the ``/docs/build/html`` folder, which can be hosted locally with ``python -m http.server <PORT-NUMBER>``. 


## Contribute

We always welcome new contributors! Feel free to open an issue or reach out to use via email if you want to collaborate.



## Attributions

This work was authored (in part) by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. 
Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views
of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, 
worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.