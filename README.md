# Fractional Turbulence Modelling 

Based on the code and theory from [this 2021](https://arxiv.org/pdf/2107.11046.pdf) publication. 

## Status of Provided Scripts 

Not all scripts provided in the repository work, with the foremost reason being that data files were not provided. Here is a listing of which can be run and reasons for those that cannot be: 

See results from initial runs in ``./script/results/``. 

- [x] Calibration Jupyter Notebook ``Calibration.ipynb`` 
- [x] Calibration script ``script_Calibration.py`` 
- [x] ``script_OnTheFlyGenerator_example.py`` 
- [ ] ``script_TestCoherence.py`` is missing the``CoherenceDataGenerator`` class ???
- [ ] ``SimpleFluctuationGenerator.py`` is missing ``fluctuation.vti``

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``fracturbulence`` into your environment. 
