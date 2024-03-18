# Deep Rapid Distortion theory Mann Turbulence model

![](https://github.com/METHODS-Group/DRDMannTurb/assets/74629347/604fcde9-41e1-4671-8c10-b1493cadfa88)


The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed by Keith et al. in [this 2021 publication](https://arxiv.org/pdf/2107.11046.pdf). 

## Installation 

Pre-compiled wheels for the package are available via ``pip install drdmannturb``. 

## Basic Usage

See the ``/examples/`` folder for baselines from the paper and for examples of the many functionalities of the package.

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``drdmannturb`` into your environment. 

We also suggest installing the local ``pre-commit`` configuration by running ``pre-commit install`` in the root directory of this repository. 

## Local Documentation Building Instructions 

Docs are in the ``/docs/`` folder. Make sure the dependencies listed in ``./requirements-docs.txt`` are installed.

Run ``make html`` to generate html pages in the ``/docs/build/html`` folder, which can be hosted locally with ``python -m http.server <PORT-NUMBER>``. 

## Citation 

If you use this software, please cite it as below.

```
@software{Izmailov_DRDMannTurb_2023,
author = {Izmailov, Alexey and Meeker, Matthew and Deskos, Georgios and Keith, Brendan},
month = mar,
title = {{DRDMannTurhb}},
version = {0.1.0},
year = {2024}
}
```

## Acknowledgements

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes. BK was supported in part by the U.S. Department of Energy Office of Science, Early Career Research Program under Award Number DE-SC0024335.