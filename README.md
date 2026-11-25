# End-to-End Customer Intelligence System for an E-commerce Store

Machine Learning course project.

**Group members**
- Muhammad Salman Saeed (2025614)
- Muhammad Hassam (2025537)

## Modules
1. **Segmentation** (unsupervised): RFM features, K-means, hierarchical clustering, GMM, PCA / t-SNE
2. **Churn prediction** (classification): Logistic Regression, Decision Tree, Random Forest, AdaBoost / GBM / XGBoost, SVM
3. **Spend (CLV) prediction** (regression): Linear, Ridge, Lasso, Gradient Boosting
4. *(optional)* Recommender

## Dataset
UCI Online Retail II (1,067,371 transaction rows, 1 Dec 2009 to 9 Dec 2011).
Download `online_retail_II.xlsx` from https://archive.ics.uci.edu/dataset/502/online+retail+ii
and put it in `data/raw/`.

## Leakage-free setup
All dates live in `src/config.py`. Features are computed only from transactions
before a cutoff date; targets (churn, next-period spend) only from the 90 days after it.
- Training table: cutoff 2011-06-01, labels from Jun - Aug 2011
- Test table:     cutoff 2011-09-01, labels from Sep - Nov 2011

## How to run
Create a virtual environment in the project folder once (Windows, from `D:\customer-intelligence`).
Use a normal python.org Python 3 for this, not the MSYS2 one:
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Then start Jupyter from the venv each time (no activation needed):
```
.venv\Scripts\python -m jupyter notebook
```
Run the notebooks in order: 01 -> 02 -> 03 -> 04 -> 05.

## Structure
```
data/raw/          original Excel file (not committed)
data/processed/    cleaned transactions + customer tables
src/               shared code (config, cleaning, features, evaluation)
notebooks/         one notebook per step / module
models/            saved models
reports/figures/   plots for the paper
reports/results/   metric tables (CSV)
```
