---
title: 'DRDMannTurb'
tags:
  - Python
  - Torch
  - Mann-model
  - wind-engineering
authors:
  - name: Alexey Izmailov
    equal-contrib: true
    affiliation: 1
  - name: Matthew Meeker
    equal-contrib: true
    affiliation: 1
  - name: Georgios Deskos
    corresponding: true
    affiliation: 2
  - name: Brendan Keith
    affiliation: 1

affiliations:
 - name: Division of Applied Mathematics, Brown University, Providence, RI, 02912, USA
   index: 1
 - name: National Wind Technology Center, National Renewable Energy Laboratory, Golden, CO, 80401, USA
   index: 2
date: 13 August 2017     # TODO -- what date to use?
bibliography: paper.bib
---

# Summary

Synthetic turbulence models (STM) are commonly used to generate realistic flow field to be used as input to a variety of models. Examples include the creation of inlet conditions for unsteady computational fluid dynamics (CFD) models, inflow wind fields to aeroelastic models of wind turbines or tall building, amongst many others. One of the advantages of STMs 
is their ability to generate fluctuations based on prescribed statistics which allows scientists and engineers to simulate and re-create environmental conditions as close as possible to the observed ones. The Mann model (@mann_spatial_1994,mann_wind_1998) which is the basis of this package, allows for the prescription of three parameters: the Kolmogorov constant multiplied by the rate of the viscous dissipation of specific turbulence kinetic energy to the two thirds, $\alpha \epsilon^{2/3}$, a turbulence length scale, $L$, and a non-dimensional parameter related to the lifetime of the eddies, $\Gamma$. A number of studies as well as international standards (e.g. IEC) have made recommendations for the values of these three parameters ($\alpha \epsilon^{2/3}$, $L$, $\Gamma$) with the goal to fit turbulence spectra or coherence functions from measurements or reference or textbook spectra. This package allows users to easily fit the Mann model to turbulence characteristics by not only modifying the aforementioned parameters, but also by adjusting the function related to the lifetime of the eddies using deep neural networks (DNN). This new approach has been found to provide excellent results with either smooth and noisy data. 

# Statement of need

`DRDMannTurb` is a package which aims to create an easy-to-use framework to automatically generate turbulent wind fields to be used by scientist and engineers in their applications domains. 

`DRDMannTurb` is completely written in Python, with computationally powerful backend packages ('numpy', 'PyTorch') being used in its implementation. Our implementation allows for easy GPU-portability using `cuda`. This is an advantage compared to previously developed software packages that have implemented the Mann model but did not provide the source code (e.g. HAWC2). Finally, `DRDMannTurb` is designed to be more general-purpose, allowing it to be applied to a bronzer range of application, as well as be more accessible and with clear documentation. 

# Results



# Package Features



# Acknowledgements

This work was authored (in part) by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes. 


# References
