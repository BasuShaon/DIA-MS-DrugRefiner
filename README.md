# DrugRefiner

## Description

Drug activity deconvolution using high-throughput screening, ultra-fast proteomics, and machine learning. 

## Environment

- Python 3.11.5 (download dependencies in local or virtual environment from 'requirements.txt') 

- R 4.3.1 / Bioconductor 3.18 (use renv to restore dependencies from 'renv.lock')

## Installation / Download

1. Download source data from FigShare (private token) 

2. Copy contents into `DrugRefiner-ML/data/`.

## Running the Code

1. Navigate into `DrugRefiner-ML/scripts/`. 

   ```sh
   cd DrugRefiner-ML/allscripts

2. Execute the scripts in alphanumerical order, starting with:

   ```sh
   Python3 fda_01_globalvarianceanalysis_250304a.py

3. View scores and outputs in `DrugRefiner-ML/data/` & `DrugRefiner-ML/figures/`