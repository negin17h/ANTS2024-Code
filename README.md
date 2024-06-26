# ANTS2024-Code

This repository contains the code and some materials used in the experimental work presented in the following paper:

**Grasshopper Optimization Algorithm (GOA): A Novel Algorithm or A Variant of PSO? ANTS2024, 2024**

## Citation
If you find the code in this repository useful for your research, consider citing our paper.

## Data

In this folder, we provided the data collected from the experiment explained in Section 5.1 of the paper.

## Figures

The figures presented in the paper in PNG and EPS formats can be found in this folder.

## Source

This folder contains the codebase for the ANTS 2024 project. The code is organized as follows:

**cls_GeneralFunctions.py:** This file contains general functions used by other classes/functions.

**cls_ObjectiveFunctions.py:** This file contains the code for all the implemented benchmark functions. Any other functions can be added to this class to be used to run the algorithms.

**cls_SwarmIntelligence.py:** This file contains the code for all the implemented SI algorithms (GOA, PSO, and its variants).

**GOA_PSO_BenchmarkFunctions_ToFile.py:** This file contains the code to run all the algorithms and save the results to be used later to plot (for the experimental setup, please refer to Section 5.1 and Table 1)

**GOA_PSO_BenchmarkFunctions_BoxPlotAll.py:** This file contains the code to read the data (from the previous steps) and plot the required boxplots.


## Statistic Test


## Appendix_BenchmarkFunctions.pdf
The mathematical definitions of the benchmark functions used in the experimental phase.

## Requirements
```
Python 3.8
NumPy 1.24.2
Pandas 1.5.3
Matplotlib 3.7.5
Seaborn 0.13.2
