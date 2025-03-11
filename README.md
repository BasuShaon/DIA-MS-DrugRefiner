# DrugRefiner

## Description

Drug activity deconvolution using high-throughput screening, ultra-fast protemics, and machine learning. 

## Requirements

- Python 3.11.5 (download dependencies in local or virtual environment from 'requirements.txt') 

- R 4.3.1 (use renv package to restore dependencies from 'renv.lock')

## Installation / Download

1. Download data from figshare

2. Copy all contents into `DrugRefiner-ML/data/`.

## Running the Code

1. Navigate into `DrugRefiner-ML/scripts/`. 

   ```sh
   cd DrugRefiner-ML/allscripts

2. Execute the scripts in alphanumerical order, starting with:

   ```sh
   Python3 fda_01_globalvarianceanalysis_250304a.py

3. View regenerated files, predictions and outputs in `DrugRefiner-ML/data/` & `DrugRefiner-ML/figures/`