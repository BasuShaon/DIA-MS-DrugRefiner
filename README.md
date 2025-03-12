# DrugRefiner

## Description

Drug activity deconvolution using high-throughput screening, ultra-fast proteomics, and machine learning. 

## Environments

- Python 3.11.5 (download dependencies in a virtual environment from 'requirements.txt') 

- R 4.3.1 + Bioconductor 3.18 (use renv to restore dependencies from 'renv.lock')

## Installation / Download

1. Clone directory in local folder

   ```sh
   git clone https://github.com/BasuShaon/DrugRefiner-ML
   cd DrugRefiner-ML

2. Download source data from FigShare (privately supplied token - 1 year emargo) 

3. Copy contents into `/data/`.

## Running the Code

1. Navigate into `/scripts/`. 

   ```sh
   cd scripts

2. Execute the scripts in alphanumerical order, starting with:

   ```sh
   Python3 fda_01_globalvarianceanalysis_250304a.py
   *Ignore modules contining device objects*

3. View scores, intermediate and output files in `/data/` & `/figures/`
