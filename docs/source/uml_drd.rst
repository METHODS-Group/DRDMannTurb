.. py:currentmodule:: drdmannturb

UML Diagram for Spectra Fitting Data Classes
============================================

Here is an UML diagram representing the interoperability between several internal classes of the package that comprise the fluctuation generator :py:class:`CalibrationProblem` and :py:class:`OnePointSpectra`. Please refer to specific class documentations for details. The following diagram is interactive -- try zooming and panning to resize for your convenience.

These interactive UML diagrams have an issue with rendering the correct arrow types in dark mode, please consider switching to light mode.

.. mermaid::
    :zoom:

    classDiagram
    direction LR
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
            - data_type : DataType
            - eddy_lifetime : EddyLifetimeType
            - learn_nu : bool
        }
        class PhysicalParameters {
            <<dataclass>>
            - L : float
            - Gamma : float
            - sigma : float
            - Uref : float
            - zref : float
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

        CalibrationProblem ..> ProblemParameters
        CalibrationProblem ..> PhysicalParameters
        OnePointSpectra ..> LossParameters
        CalibrationProblem ..> OnePointSpectra
        OnePointSpectra ..>  ProblemParameters
        OnePointSpectra  ..> NNParameters
