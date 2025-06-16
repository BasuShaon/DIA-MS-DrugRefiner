# HT-MS-DrugRefiner

Drug activity deconvolution using high-throughput screening, ultra-fast proteomics, and machine learning. 

## Project Structure

```
project_root/      
│     
│-- code/                 # main folder with all scripts
│   ├── device_gradientboostingmachine.py
│   ├── device_summarystatistics.py    
│   ├── fda_01_globalvarianceanalysis_250324a.py
│   ├── fda_02_limma_drug_250304a.R
│   ├── fda_03_de_pca_250304a.py
│   ├── fda_04_de_gsea_250304a.py
│   ├── protacs_01_globalvarianceanalysis_250304a.py
│   ├── protacs_02_azmetadata_cluster_250306a.py
│   ├── protacs_03_azmetadata_dendrogram_250306a.R
│   ├── protacs_04_limma_metadataconstructor_250304a.py
│   ├── protacs_05_limma_cluster_0p1_250305a.R
│   ├── protacs_06_limma_cluster_1p0_250305a.R
│   ├── protacs_07_limma_cluster_10_250305a.R
│   ├── protacs_08_limma_drug_250305a.R
│   ├── protacs_09_de_pca_250305a.py
│   ├── protacs_10_de_gsea_250305.py
│   ├── protacs_11_de_stringnetworkenrich_250305a.py
│   ├── protacs_12_de_regression_250305a.py
│   ├── protacs_13_split_260311a.py
│   ├── protacs_14_split_limma_250306a.R
│   ├── protacs_15_split_enrich_250306a.py
│   ├── protacs_16_ML_brutalgrid_0p01learn_retrain_250306a.py
│   ├── protacs_17_ML_finalmodel_predictions_250307.py
│   ├── protacs_18_ML_dotplot_250307a.R
│   ├── protacs_19_ML_dotplot_analog_250307a.R
│   ├── protacs_20_stats_wetlab_250307a.py
│   ├── protacs_21_safetyscore_distributions_250522a.ipynb
│ 
│-- data/             
│   ├── README.md         # instructions on sourcedata aquisition
│
│-- figures/              # output folder to regenerate all figs
│   ├── README.md           
│ 
│-- scoring_models/             
│   ├── final_model_250305a.json
│   ├── final_calibrated_model_250305a.pkl
│
│-- requirements.txt      # Python dependencies for virtual / local env
│-- renv.lock             # R environment lockfile
│-- .gitignore  
│-- README.md  

```
## Setup Instructions

### Environments

- Python 3.11.5 (download dependencies in a virtual environment from 'requirements.txt') 

- R 4.3.1 + Bioconductor 3.18 (use renv to restore dependencies from 'renv.lock')

### Installation / Download

1. Clone directory in local folder

   ```sh
   git clone https://github.com/BasuShaon/HT-MS-DrugRefiner.git
   cd HT-MS-DrugRefiner
   pip install -r requirements.txt

2. Download source data from FigShare (privately shared URL) 

3. Copy contents into `/data`.

### Running the Code

1. Navigate into `/scripts`. 

   ```sh
   cd scripts

2. Execute the scripts in alphanumerical order, starting with:

   ```sh
   Python3 fda_01_globalvarianceanalysis_250304a.py
   Rscript fda_02_limma_drug_250304a.R
   ...

   *Ignore python modules containing device objects*

3. View scores, intermediate and output files in `/data` & `/figures`
