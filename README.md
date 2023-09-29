# Deep Rapid Distortion theory Mann Turbulence model

The code is based on the original work of Jacob Mann in [1994](https://doi.org/10.1017/S0022112094001886) and [1998](https://doi.org/10.1016/S0266-8920(97)00036-2)
as well as in the deep-learning enhancement developed by Keith et al and presented in [this 2021 publication](https://arxiv.org/pdf/2107.11046.pdf). 

## Baselines

See the ``/examples/`` folder for baselines from the paper and for several functionalities of the package.

### Noisy Data (Experiment 2) 

Using a 4-layer 32-neuron FC neural network with a CosineAnnealer, these are the results after 10 epochs, which have a minimum loss value of  
``loss = 0.14850306846012104``, as compared to 100 epochs yielding ``loss = 0.20783722970539706`` before. 
![image](https://github.com/mjachi/WindGenerator/assets/74629347/838b8767-28c8-46cd-b349-5e2925255462)

## Development Installation Instructions 

This package is provided as a locally ``pip``-installable package. It is recommended that Python 3.9 is used in your local environment. 

A Conda spec-file is included, to install a functioning environment, run ``conda create --name ENV_NAME_HERE --file spec-file.txt``. Then run ``pip install -e .`` to load ``drdmannturb`` into your environment. 

## Doc Building Instructions 

Docs are in the ``/docs/`` folder. Make sure the following dependencies are installed:

Run ``make html`` to generate html pages in the ``/docs/build/html`` folder, which can be hosted locally with ``python -m http.server <PORT-NUMBER>``. 
```mermaid
---
title: Phase 1 - CalibrationProblem Class System
---
classDiagram
    class CalibrationProblem {
        <<class>>
        - output_directory : Path
        update_parameters() 
        eval()
        eval_grad() 
        calibrate()
        plot()
        plot_loss()
    }
    class NNParameters {
        <<dataclass>>
        - nlayers : int
        - input_size : int
        - hidden_layer_size : int
        - hidden_layer_sizes : list[int]
        - activations : list[str]
        - output_size : int 
    }
    class ProblemParameters {
        <<dataclass>>
        - learning_rate : float
        - tol : float 
        - nepochs : int 
        - init_with_noise : bool 
        - noise_magnitude : float
        - fg_coherence : bool 
        - data_type : DataType
        - eddy_lifetime : EddyLifetimeType 
        - power_spectra : PowerSpectraType 
        - learn_nu : bool
    }
    class PhysicalParameters {
        <<dataclass>>
        - L : float 
        - Gamma : float 
        - sigma : float 
        - Uref : float 
        - zref : float 
        - Iref : float 
        - sigma1 : float 
        - Lambda1 : float 
        - z0 : float 
        - Lambda1 : float 
        - z0: float 
        - ustar : float 
        - domain : torch.Tensor
    }
    class LossParameters {
        <<dataclass>>
        - alpha_pen : float 
        - alpha_reg : float 
        - beta_pen : float 
    }
    class OnePointSpectra { 
        <<class>>
        - grid_k2 : torch.Tensor 
        - grid_k3 : torch.Tensor 
        - meshgrid23 : torch.Tensor 
        - logLengthScale : nn.Parameter
        - logTimeScale : nn.Parameter 
        - logMagnitude : nn.Parameter 
        - tauNet : nn.Module**
        exp_scales()
        forward(k1_input)
        EddyLifetime(k)
        PowerSpectra()
        quad23()
        get_div()
    }

    CalibrationProblem o-- NNParameters
    CalibrationProblem o-- ProblemParameters
    CalibrationProblem o-- PhysicalParameters
    CalibrationProblem o-- LossParameters
    ProblemParameters o-- OnePointSpectra
    NNParameters o-- OnePointSpectra
```