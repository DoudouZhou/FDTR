# Federated Dynamic Treatment Regime (FDTR)
Code for the paper Federated Offline Reinforcement Learning

## Overview

Running `simulations.py` will 

1) Generate a training dataset using random behavior policy
2) Train an FDTR policy
3) Train LDTR, LDTR (MV), and 3 different Q-learning policies (see the paper for details)
4) Evaluate the policies on K hospital sites

Results are saved as a CSV file and estimated parameters from Algorithm 1 are saved as a pickle file which contains a dictionary.

### Function:

To begin the process `simulations.py` with the following options:
```
python simulations.py Hs_dim ${1} Ps_dim ${2} a_No ${3} H ${4} episodes_No ${5} K ${6}
```
where
- Hs_dim: the hospital-level state dimension
- Ps_dim: the patient-level state dimension
- a_No: cardinality of action space
- H: episode length 
- episodes_No: sample size
- K: Number of hospital sites

There are three other files:
* `utils.py` contains all functions
* `utils_sepsis.py` contains aditional functions for the sepsis data analysis
* `sepsis_FDTR.py` contains code to run the analysis using the MIMIV-IV data set which is publicly available
