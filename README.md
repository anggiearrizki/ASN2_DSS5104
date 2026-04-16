# DSS5104 Machine Learning and Predictive Modelling  
## Assignment 2: Fraud Detection Project

## ===== EXECUTION ORDER =====
Run the notebooks in the following order to generate the final results. Each notebook's inputs and outputs are specified.

1. **1_fe.ipynb (Feature Engineering)**  
   - **Inputs**: Raw IEEE-CIS data files (`data_raw/IEEE-CIS/ieee-fraud-detection/train_transaction.csv`, `train_identity.csv`)  
   - **Outputs**: `data/iceee_baseline.parquet` (baseline cleaned data), `data/iceee_feature.parquet` (engineered features)  
   - **Purpose**: Data preprocessing, memory optimization, feature engineering  

2. **2_ml.ipynb (Machine Learning Models)**  
   - **Inputs**: `data/iceee_baseline.parquet`, `data/iceee_feature.parquet`  
   - **Outputs**: `results/feature_impact_results.csv` (baseline vs. engineered comparison), `results/all_ml_probs.csv` (model probabilities for LGBM, XGB, ISO, LOF)  
   - **Purpose**: Train and evaluate classical ML models  

3. **3_dl.ipynb (Deep Learning Models)**  
   - **Inputs**: `data/iceee_feature.parquet`  
   - **Outputs**: `results/dl_results.csv` (probabilities for AE semi-supervised, AE supervised, MLP)  
   - **Purpose**: Train and evaluate deep learning models (Autoencoders, MLP)  

4. **4_result.ipynb (Results Analysis and Comparison)**  
   - **Inputs**: `results/all_ml_probs.csv`, `results/dl_results.csv`, `results/feature_impact_results.csv`  
   - **Outputs**: `report/final_model_comparison_table.csv` (performance ranking), `report/top_10_false_positives.csv`, `report/top_10_false_negatives.csv`, `report/final_pr_curves.png`  
   - **Purpose**: Combine results, cost-sensitive evaluation, error analysis  

**Note**: Notebooks 2_ml.ipynb and 3_dl.ipynb can be run in parallel after 1_fe.ipynb completes.

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
- `data_raw/IEEE-CIS/ieee-fraud-detection/train_transaction.csv`
- `data_raw/IEEE-CIS/ieee-fraud-detection/train_identity.csv`

### PaySim
Place in:
Required file:
- `data_raw/PaySim/PS_20174392719_1491204439457_log.csv`

## ===== NOTES =====
- A Kaggle account is required to download the datasets  
- Processed files (e.g., `merged_data.csv`) are generated locally and not committed


