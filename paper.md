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
is their ability to generate fluctuations based on prescribed statistics which allows scientists and engineers to simulate and re-create environmental conditions as close as possible to the observed ones. The Mann model \cite{mann_spatial_1994,mann_wind_1998} which is the basis of this package, allows for a prescription of three parameters: the Kolmogorov constant multiple with the rate of the viscous dissipation of specific turbulence kinetic energy to the two thirds, $\alpha \epsilon^{2/3}$, the length scale, $L$, the non-dimensional parameter related to the lifetime of the eddies, $\Gamma$.  

# Statement of need

TEMPLATE SECTION 

# Figures

TEMPLATE SECTION 

# Acknowledgements

TEMPLATE SECTION 

# References
