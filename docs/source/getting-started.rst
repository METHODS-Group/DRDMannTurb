.. py:currentmodule:: drdmannturb

Getting Started
===============

.. mermaid:: 
    :zoom:

    classDiagram
        class CalibrationProblem {
            <<class>>
            - output_directory : Path
            - device : str
            init_device()
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
        CalibrationProblem o-- OnePointSpectra
        OnePointSpectra o--  ProblemParameters
        OnePointSpectra  o-- NNParameters
