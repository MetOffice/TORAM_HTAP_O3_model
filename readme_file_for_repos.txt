## This file is associated with the python code of the HTAP Parameterisation developed by Steven Turnock (Met Office Hadley Centre) and based on that from Oliver Wild (Lancaster University)

There are four main python scripts associated with this repository:

1. HTAP1_and_2_combined_O3_para_model_for_repos.py - Main script for parameterisation containing calls to other functions/files

2. htap_functions.py - General functions called from the main script  

3. H1_funcs_for_REPOS.py - Functions specifically used for HTAP1 models called from the main script

4. H2_funcs_for_REPOS.py - Functions specifically used for HTAP2 models called from the main script

## Also included are two example fractional emission change input files ##

## In addition to these scripts the following input files are also required to be able to use the parameterisation (available upon request):

1. Pre-processed HTAP1 model files (.nc) with O3 response to 20% emission perturbation experiments

2. Pre-processed HTAP2 model files (.nc) with O3 response to 20% emission perturbation experiments

3. 2010 Baseline file (.nc)

4. ACCMIP IDL .sav file contraining relationship of O3 burden to Radiative forcing

5. HTAP2 receptor region definition file (.nc) 

6. Scenario speific fractional emission change files (.txt) set up in specific format so that parameterisation can read

