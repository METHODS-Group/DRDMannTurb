# Fractional Turbulence Modelling 

Based on the code and theory from [this 2021](https://arxiv.org/pdf/2107.11046.pdf) publication. 

## Status of Provided Scripts 

Most provided scripts now work out of the box. Here is a listing of which can be run: 

See results from initial runs in ``./script/results/``. 

- [x] Calibration Jupyter Notebook ``Calibration.ipynb`` 
- [x] Calibration script ``script_Calibration.py`` 
- [x] ``script_OnTheFlyGenerator_example.py`` 
- [x] ``script_TestCoherence.py`` 
- [ ] ~~``new_script_TestCoherence.py`` is missing the``CoherenceDataGenerator`` class ??? (replaced by old ``script_TestCoherence.py``)~~
- [x] ``SimpleFluctuationGenerator.py`` 

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``fracturbulence`` into your environment. 

## Optimization  

TODO: write a ``CalibrationLite`` class that subclasses from ``nn.Module`` that has bare minimum functionality that matches the current ``Calibration`` class 

- [ ] Investigate unnecessary ``np`` calls and other CPU-only libraries that contribute to high VRAM volatility (averaging 25%) 
- [ ] The ``MannEddyLifetime`` function requires a ``scipy.special`` call, which incurs a communication between CPU and GPU; this may require something like the (now defunct) [torch-Cephes library](https://github.com/deepmind/torch-cephes)...  
