# Deep Rapid Distortion theory Mann Turbulence model (DRDMannTurb)

![](https://github.com/METHODS-Group/DRDMannTurb/blob/main/.github/assets/anim_gh.gif)

`DRDMannTurb` is a GPU-accelerated, data-driven Python framework for
the automatic fitting of spectra data and generation of synthetic turbulent wind fields. This package is intended to be used by
wind engineers in applications requiring rapid simulation of realistic wind turbulence. It is based off of the Deep Rapid Distortion models presented in [Keith, Khristenko, Wohlmuth (2021)](https://arxiv.org/pdf/2107.11046.pdf) and the Mann's original work in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2). The data-driven functionalities are GPU-accelerated with [PyTorch](https://pytorch.org/docs/stable/index.html).

## Installation and Getting Started

The following sections will guide you through the installation of DRDMannTurb and its dependencies. The following instructions should work on any operating system (OS) that is supported by [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html), including Windows, macOS, and Linux. 

1. Clone DRDMannTurb from its [GitHub repository](https://github.com/METHODS-Group/DRDMannTurb).
Enter the directory and check out the branch of choice.
The latest development version will be available under the branch ``develop``.

```shell
git clone https://github.com/METHODS-Group/DRDMannTurb.git
cd DRDMannTurb
git checkout main
```

2. Create an Anaconda environment (called `drdmannturb_env` here) for installing the package.
Use the default environment specs in `env_drdmannturb.yml` to create it.
Then activate the environment:

```shell
conda env create -n drdmannturb_env -f requirements/env_drdmannturb.yml
conda activate drdmannturb_env
```

3. Install the local DRDMannTurb source files as a Python package using ``pip`` in the root directory:

```shell
python -m pip install -e ./
```

See the ``/examples/`` folder for baselines from the paper and demonstrations of the many functionalities of the package. For more detail on the examples themselves,
see the inline comments or the corresponding page in the Examples section of our
documentation

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``drdmannturb`` into your environment. 

We also suggest installing the local ``pre-commit`` configuration by running ``pre-commit install`` in the root directory of this repository. 

## Local Documentation Building Instructions 

Docs are in the ``/docs/`` folder. Make sure the dependencies listed in ``./requirements-docs.txt`` are installed.

Run ``make html`` to generate html pages in the ``/docs/build/html`` folder, which can be hosted locally with ``python -m http.server <PORT-NUMBER>``. 

## Attribution

