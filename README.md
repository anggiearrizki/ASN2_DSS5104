# DSS5104 Machine Learning and Predictive Modelling  
## Assignment 2: Fraud Detection Project

## ===== DATASETS =====
### Primary Dataset: IEEE-CIS Fraud Detection
- Source: https://www.kaggle.com/competitions/ieee-fraud-detection/data  
- Contains transaction, identity, and behavioral features  
- Used for feature engineering and main model evaluation  

### Secondary Dataset: PaySim
- Source: https://www.kaggle.com/datasets/ealaxi/paysim1  
- Simulated financial transactions dataset  
- Used for comparison and validation  

## ===== DATA SETUP =====
The raw data files are not included in this repository due to size constraints.  
Please download them manually and place them in the correct folders.

### IEEE-CIS
Place in:
Required files:
- `transaction.csv`
- `identity.csv`

### PaySim
Place in:
Required file:
- `PS_20174392719_1491204439457_log.csv`

## ===== NOTES =====
- A Kaggle account is required to download the datasets  
- Processed files (e.g., `merged_data.csv`) are generated locally and not committed



### ---------------- understanding the notebooks + their output -----------------
01_ieee_eda.ipynb
  -> data/ieee_clean.parquet

02_ieee_features.ipynb
  -> data/ieee_baseline.parquet
  -> data/ieee_featured.parquet

01_paysim_eda.ipynb
  -> data/paysim_clean.parquets

02_paysim_features.ipynb
  -> data/paysim_baseline.parquet
  -> data/paysim_featured.parquet

03_classical_ml.ipynb   (run once for ieee, once for paysim)
  -> models/{dataset}_xgb_featured.pkl
  -> models/{dataset}_xgb_baseline.pkl
  -> models/{dataset}_iso_forest.pkl
  -> models/{dataset}_lof.pkl
  -> results/{dataset}_classical_metrics.json
  -> results/{dataset}_*_test_preds.npy

04_deep_learning.ipynb  (run once for ieee, once for paysim)
  -> models/{dataset}_ae_semi.pth
  -> models/{dataset}_ae_sup.pth
  -> models/{dataset}_vae_semi.pth
  -> models/{dataset}_vae_sup.pth
  -> models/{dataset}_nn.pth
  -> results/{dataset}_dl_metrics.json
  -> results/{dataset}_*_test_preds.npy

05_results.ipynb
  -> loads metrics + saved preds
  -> PR curves
  -> cost analysis
  -> error analysis
  -> cross-dataset comparison