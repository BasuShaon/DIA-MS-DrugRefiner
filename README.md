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
   Rscript fda_02_limma_drug_250304a.R      
   Rscript fda_02_limma_drug_250304a.R
   Python3 fda_03_de_pca_250304a.py
   Python3 fda_04_de_gsea_250304a.py
   Python3 protacs_01_globalvarianceanalysis_250304a.py
   Python3 protacs_02_azmetadata_cluster_250306a.py
   Rscript protacs_03_azmetadata_dendrogram_250306a.R
   Python3 protacs_04_limma_metadataconstructor_250304a.py
   Rscript protacs_05_limma_cluster_0p1_250305a.R
   Rscript protacs_06_limma_cluster_1p0_250305a.R
   Rscript protacs_07_limma_cluster_10_250305a.R
   Rscript protacs_08_limma_drug_250305a.R
   Python3 protacs_09_de_pca_250305a.py
   Python3 protacs_10_de_gsea_250305.py
   Python3 protacs_11_de_stringnetworkenrich_250305a.py
   Python3 protacs_12_de_regression_250305a.py
   Python3 protacs_13_split_260311a.py
   Rscript protacs_14_split_limma_250506a.R
   Python3 protacs_15_split_enrich_250506a.py
   Python3 protacs_16_ML_brutalgrid_0p01learn_retrain_250506a.py
   Python3 protacs_17_ML_finalmodel_predictions_250507.py
   Rscript protacs_18_ML_dotplot_250507a.R
   Rscript protacs_19_ML_dotplot_analog_250507a.R
   Python3 protacs_20_stats_wetlab_250507a.py

   *Ignore modules containing device objects*

3. View scores, intermediate and output files in `/data/` & `/figures/`
