.. py:currentmodule:: drdmannturb

UML Diagram for Fluctuation Field Generation
============================================

Here is an UML diagram representing the interoperability between several internal classes of the package that comprise the fluctuation generator :py:class:`FluctuationFieldGenerator`. Please refer to specific class documentations for details. The following diagram is interactive -- try zooming and panning to resize for your convenience.

These interactive UML diagrams have an issue with rendering the correct arrow types in dark mode, please consider switching to light mode.

.. mermaid::
    :zoom:

    classDiagram
    direction LR
    FluctuationFieldGenerator ..> VectorGaussianRandomField
    GaussianRandomField ..> Covariance

    VectorGaussianRandomField --|> GaussianRandomField

    subgraph cov
    direction TB
    MannCovariance --|> Covariance
    VonKarmanCovariance --|> Covariance
    NNCovariance --|> Covariance
    end

    FluctuationFieldGenerator : +np.ndarray total_fluctuation
    FluctuationFieldGenerator : +Callable log_law
    FluctuationFieldGenerator : +int seed
    FluctuationFieldGenerator : +Covariance Covariance
    FluctuationFieldGenerator : +VectorGaussianRandomField RF

    FluctuationFieldGenerator: -_generate_block()
    FluctuationFieldGenerator: +generate()
    FluctuationFieldGenerator: +normalize()
    FluctuationFieldGenerator: +save_to_vtk()
    class VectorGaussianRandomField{
      +int vdim
      +tuple DomainSlice
    }
    class Covariance{
      +int ndim
      +eval()
    }
    class GaussianRandomField{
    +np.RandomState prng
    +Callable Correlate
    +Callable Covariance

    +setSamplingMethod()
    +sample_noise()
    +sample()
    +reseed()
    }
    class VonKarmanCovariance{
    +float length_scale
    +float E0
    +precompute_Spectrum()
    }
    class MannCovariance{
    +float length_scale
    +float E0
    +float Gamma
    +precompute_Spectrum()
    }
    class NNCovariance{
    +float length_scale
    +float E0
    +float Gamma
    +float h_ref
    +OnePointSpectra ops
    +precompute_Spectrum()
    }
