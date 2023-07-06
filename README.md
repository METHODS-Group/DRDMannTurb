# Fractional Turbulence Modelling 

Based on the code and theory from [this 2021](https://arxiv.org/pdf/2107.11046.pdf) publication. 

## Baselines

### Noisy Data (Experiment 2) 

Using a 4-layer 32-neuron FC neural network with a CosineAnnealer, these are the results after 10 epochs, which have a minimum loss value of  
``loss = 0.14850306846012104``, as compared to 100 epochs yielding ``loss = 0.14850306846012104`` before. 
![image](https://github.com/mjachi/WindGenerator/assets/74629347/838b8767-28c8-46cd-b349-5e2925255462)


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

