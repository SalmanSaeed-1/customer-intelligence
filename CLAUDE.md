# CLAUDE.md: Project context for Claude Code

> This file holds **everything decided so far** about our ML course project.
> Claude Code reads this file automatically at the start of every session.
> Read all of it before writing code. If something here conflicts with a request, ask before changing it.

---

## 1. Who we are and what this is

| Item | Detail |
|---|---|
| Course | Machine Learning (university course project) |
| Project title | **End-to-End Customer Intelligence System for an E-commerce Store** |
| Group member 1 | Muhammad Salman Saeed (2025614) |
| Group member 2 | Muhammad Hassam (2025537) |
| Time available | **5 weeks** (project started mid-November 2026), alongside other courses |
| Project location on PC | `D:\customer-intelligence\` (Windows) |
| Language / tools | Python 3, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, Jupyter |
| Deadline | _fill in: ____________ |

**One-sentence pitch:** Using one real e-commerce transaction dataset, we build a single ML pipeline that (1) segments customers, (2) predicts which customers will churn and (3) predicts how much each customer will spend next. A **time-based split** (cutoff date) keeps all of it free of data leakage.

**Key interview / viva talking point:** we use a **time-based split with a cutoff date** so features come only from the past and targets only from the future. A random split would leak future information into training.

---

## 2. Decisions already made (do not change without asking)

1. **Three core modules**, cut down from the original four because of the 5-week limit:
   - Module 1: Customer segmentation (unsupervised)
   - Module 2: Churn prediction (classification)
   - Module 3: Spend / CLV prediction (regression)
   - Module 4: Recommender. **OPTIONAL, only if time permits.** It is listed as an "optional extension" in the proposal.
2. **Dataset:** UCI Online Retail II, one dataset for all modules.
3. **Time-based split with two cutoffs**, giving a separate training snapshot and test snapshot (see Section 5).
4. **Label window = 90 days** after each cutoff, so both snapshots have the same-length target period.
5. **Depth over breadth:** for each module, tune and analyse 1 or 2 models properly. The other algorithms are baselines in a comparison table.
6. **Modules are linked:** the segment label from Module 1 is used as a feature in Modules 2 and 3. Churn probability × predicted spend gives a "retention priority" score.
7. **Shared code lives in `src/`.** Notebooks import from it, and all dates, paths and the seed live only in `src/config.py`.
8. `RANDOM_STATE = 42` everywhere.

---

## 3. What has been done so far (status log)

- [x] Project idea chosen and reviewed
- [x] Scope cut from 4 modules to 3 (recommender optional)
- [x] **Project proposal written** (2-page Word file, in the professor's format: Title, Group Members, Problem Statement, Motivation, Background, Related Work, Proposed Methodology, References). Saved as `reports/proposal/Project_Proposal.docx`.
- [x] Project folder structure designed and created (zip extracted to `D:\customer-intelligence\`)
- [x] `src/config.py` written (paths, cutoffs, seed, module colours)
- [x] 6 notebook skeletons created, each with a step checklist and setup cell
- [~] Intro figures for the paper: Figure 1 (system overview diagram) drafted; Figures 2 and 3 planned (see Section 13)
- [x] Download dataset into `data/raw/` (19 Nov 2026; checked: 2 sheets, 8 columns, 525,461 + 541,910 = 1,067,371 rows)
- [x] Python environment: `.venv` (Python 3.14, pandas 3.1, scikit-learn 1.9, xgboost 3.4) (21 Nov 2026)
- [x] `src/cleaning.py` + notebook 01 (25 Nov 2026): 776,577 clean rows, 5,852 customers; step table in `reports/results/cleaning_steps.csv`; 5 EDA figures `reports/figures/eda_*.png/.pdf`. See "Findings from notebook 01" in Section 7.
- [ ] `src/features.py` + notebook 02
- [ ] Notebook 03 segmentation
- [ ] Notebook 04 churn
- [ ] Notebook 05 CLV
- [ ] (optional) Notebook 06 recommender
- [ ] Final report / paper + presentation

> Update this checklist as work gets done.

---

## 4. Dataset: UCI Online Retail II

- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (file `online_retail_II.xlsx`, ~43.5 MB, licence CC BY 4.0)
- A UK-based, registered, non-store online retailer that sells mainly unique all-occasion giftware. **Many customers are wholesalers.**
- **1,067,371 rows**, 8 columns, transactions from **01 Dec 2009 to 09 Dec 2011**
- **5,942 unique customer IDs**, **243,007 rows with missing Customer ID**, 43 countries (mostly UK)
- The Excel file has **two sheets**: `Year 2009-2010` and `Year 2010-2011`. **Load both and concatenate them** (`pd.read_excel(path, sheet_name=None)` returns both).
- Reading the Excel file is slow (~1–2 min). Read it once in notebook 01 and save it as parquet.

| Raw column | Meaning | Our name (snake_case) |
|---|---|---|
| `Invoice` | 6-digit invoice no.; **starts with "C" = cancellation** | `invoice` |
| `StockCode` | 5-digit product code (some have a letter suffix, e.g. `85123A`); special non-product codes exist | `stock_code` |
| `Description` | product name (some missing) | `description` |
| `Quantity` | units per line; negative for returns/cancellations | `quantity` |
| `InvoiceDate` | date + time of the invoice | `invoice_date` |
| `Price` | unit price in GBP (£) | `price` |
| `Customer ID` | 5-digit customer number (loads as float, convert to int) | `customer_id` |
| `Country` | customer's country | `country` |
| (new) | `quantity * price` | `revenue` |

---

## 5. Leakage-free time split (most important design)

```
Dec 2009 ───────────────────────── Jun 2011 ─── Sep 2011 ─── Dec 2011
|<──────── TRAIN features ────────>|<─ TRAIN labels (90 d) ─>|
|<──────────────── TEST features ─────────────>|<─ TEST labels (90 d) ─>|
```

| | Features computed from | Labels computed from |
|---|---|---|
| **Train table** | `invoice_date < 2011-06-01` | `2011-06-01 <= invoice_date < 2011-08-30` (90 days) |
| **Test table** | `invoice_date < 2011-09-01` | `2011-09-01 <= invoice_date < 2011-11-30` (90 days) |

Rules:
- **Customer population for a snapshot** = customers with at least one purchase before that cutoff.
- **Features ONLY from before the cutoff.** This filter lives inside `src/features.py`, so nobody can forget it.
- **Labels ONLY from the 90-day window after the cutoff.**
- Train models on the train table and evaluate on the test table. Hyper-parameter tuning uses cross-validation **inside the train table only**.
- Any scaler, encoder or clustering model is **fit on the train table only**, then applied to the test table (use sklearn `Pipeline`).
- The same customer can appear in both tables (two snapshots at different times). That is expected and fine: the test labels (Sep–Nov) are never seen during training.
- Note: the proposal text says "holdout Sep–Dec 2011". The final design uses a fixed 90-day window (Sep–Nov), a minor refinement.

**Why not a random split?** With a random split, rows from after the cutoff end up in training features, so the model "sees the future" and scores look unrealistically good.

---

## 6. Folder structure

```
D:\customer-intelligence\
├── CLAUDE.md                  # this file
├── README.md
├── requirements.txt           # pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl pyarrow jupyter
├── .gitignore                 # ignores data/raw/*, data/processed/*, models/*
├── data/
│   ├── raw/                   # online_retail_II.xlsx (never edited)
│   └── processed/
│       ├── transactions_raw.parquet          # cache of the raw Excel (both sheets), made by load_raw()
│       ├── transactions_clean.parquet        # output of 01
│       ├── cancellations.parquet             # output of 01: one row per cancelled invoice (for cancel_rate)
│       ├── customer_table_train.parquet      # output of 02 (+ segment col from 03)
│       └── customer_table_test.parquet       # output of 02 (+ segment col from 03)
├── src/
│   ├── __init__.py
│   ├── config.py              # DONE: paths, dates, seed, colours
│   ├── cleaning.py            # DONE: load_raw(), clean_transactions(), get_cancellations()
│   ├── features.py            # TODO: build_rfm(), build_extra_features(), build_targets(), build_customer_table()
│   └── evaluation.py          # STARTED: set_plot_style(), save_fig(), save_table(); metric tables/plots TODO
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_segmentation.ipynb
│   ├── 04_churn_prediction.ipynb
│   ├── 05_clv_prediction.ipynb
│   └── 06_recommender_optional.ipynb
├── models/                    # saved fitted models (.pkl / joblib)
└── reports/
    ├── figures/               # all plots, PNG 300 dpi
    ├── results/               # metric tables as CSV
    ├── proposal/              # Project_Proposal.docx
    └── final_report/          # paper + slides
```

### `src/config.py` (already written, current contents)
```python
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"

RAW_FILE = DATA_RAW / "online_retail_II.xlsx"
RAW_CACHE = DATA_PROCESSED / "transactions_raw.parquet"
CLEAN_FILE = DATA_PROCESSED / "transactions_clean.parquet"
CANCEL_FILE = DATA_PROCESSED / "cancellations.parquet"
TRAIN_TABLE = DATA_PROCESSED / "customer_table_train.parquet"
TEST_TABLE = DATA_PROCESSED / "customer_table_test.parquet"

DATA_START = pd.Timestamp("2009-12-01")
DATA_END = pd.Timestamp("2011-12-09")
HOLDOUT_DAYS = 90
TRAIN_CUTOFF = pd.Timestamp("2011-06-01")
TEST_CUTOFF = pd.Timestamp("2011-09-01")

RANDOM_STATE = 42
COLORS = {"segmentation": "#2a78d6", "churn": "#eb6834", "clv": "#1baf7a",
          "neutral": "#8c8c8c", "ink": "#1f1f1f"}
# (also creates the output folders if missing)
```

### Notebook setup cell (first code cell of every notebook)
```python
import sys
sys.path.append("..")   # lets the notebook import from src/
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src import config
np.random.seed(config.RANDOM_STATE)
```

### Data flow
1. **01** reads the raw Excel (both sheets), cleans it and saves `transactions_clean.parquet`.
2. **02** builds the customer table at `TRAIN_CUTOFF` and at `TEST_CUTOFF` and saves both.
3. **03** fits clustering on train, predicts segments for test, and adds a `segment` column to both tables.
4. **04** and **05** load both tables (with segments). They differ only in the target column.

---

## 7. Cleaning rules (`src/cleaning.py`, notebook 01)

Apply in this order, and **print the row count after each step** (this becomes a table in the report):
1. Load both sheets and concatenate them; rename columns to snake_case.
2. Drop exact duplicate rows.
3. Drop cancellations: `invoice` starts with `"C"`. (Optionally, first count cancellations per customer as a feature, `cancel_rate`.)
4. Drop rows with missing `customer_id`, then convert it to int.
5. Drop `quantity <= 0` and `price <= 0`.
6. Drop non-product stock codes, e.g. `POST, DOT, M, D, C2, BANK CHARGES, CRUK, PADS, AMAZONFEE, ADJUST, TEST001, TEST002, S, B`. A general rule is to keep codes that start with a digit (`stock_code.str.match(r"^\d")`), then check what was removed.
7. Add `revenue = quantity * price`.
8. Parse `invoice_date` as datetime.
9. Save to `config.CLEAN_FILE` (parquet).

EDA plots for notebook 01 (save to `reports/figures/`): monthly revenue over time, monthly active customers, top 10 countries, revenue-per-customer distribution (log scale), orders-per-customer distribution.

### Findings from notebook 01 (25 Nov 2026)
- Row counts: raw 1,067,371 → duplicates −34,335 → cancellations −19,104 → missing customer −234,437 → qty/price ≤ 0 −70 → non-product codes −2,848 → **776,577 clean rows (72.8%)**. 5,852 customers, 36,594 invoices, 4,619 products, 41 countries, £17.07M.
- **The two Excel sheets overlap on 1–9 Dec 2010** (~22,800 of the duplicates). The duplicate rule is what stops those days being counted twice.
- Pandas 3: `invoice` and `stock_code` mix ints and strings in the Excel file, so `load_raw()` casts them to `str`; without this, `.str` filters silently skip the numeric rows and parquet refuses to save.
- The digit rule only removes one real product (`SP1002`, 2 rows). `DCGS…` and `gift_0001_…` codes never have a customer ID.
- **Strong seasonality:** Sep–Nov is the yearly peak (>£1.1M/month in Nov), Jun–Aug is quiet (~£0.6M). Train labels fall in the quiet season and test labels in the peak, so expect lower churn and higher spend in the test table (a distribution shift to discuss in the report).
- 91% of customers are UK; 28% of customers ordered only once; revenue per customer: median £856, max ~£581k.
- **OPEN DECISION:** two huge purchases were cancelled ~15 min later (`541431`/`C541433`, customer 12346, 18 Jan 2011, £77k; `581483`/`C581484`, customer 16446, 9 Dec 2011, £168k). The rules drop the cancellation lines but keep the purchases. The Dec 2011 one is outside every window; the Jan 2011 one inflates customer 12346's `monetary` in both tables. Option: also drop purchase lines with an exact matching cancellation (same customer, stock code, opposite quantity). Not applied; needs both members to agree.
- `cancel_rate` (optional) must be computed in `features.py` from `config.CANCEL_FILE` using only cancellations **before** the cutoff (counting over all dates would leak).
- EDA figures are named `eda_*.png/.pdf` (not `figXX_…`), because their order in the paper isn't decided yet.

---

## 8. Features and targets (`src/features.py`, notebook 02)

All features are per customer and use **only** `tx[tx.invoice_date < cutoff]`.

| Feature | Definition |
|---|---|
| `recency_days` | days from last purchase to the cutoff |
| `frequency` | number of distinct invoices |
| `monetary` | total revenue |
| `avg_order_value` | monetary / frequency |
| `tenure_days` | days from first purchase to the cutoff |
| `n_unique_products` | distinct stock codes bought |
| `avg_basket_size` | mean number of items (quantity) per invoice |
| `mean_interpurchase_days` | mean gap between consecutive invoices (NaN if 1 order: fill with tenure or a large value, and add `is_single_order` flag) |
| `revenue_last_90d` | revenue in the 90 days before the cutoff |
| `frequency_last_90d` | invoices in the 90 days before the cutoff |
| `is_uk` | 1 if country == United Kingdom |
| `cancel_rate` *(optional)* | cancelled invoices / all invoices (needs cancellations counted before they are dropped) |
| `segment` | added by Module 1 |

Targets use **only** `cutoff <= invoice_date < cutoff + 90 days`:
- `churned` = 1 if the customer has **no** invoice in the window, else 0
- `next_period_spend` = total revenue in the window (0 if none)

Function signatures:
```python
build_rfm(tx, cutoff) -> DataFrame[customer_id, recency_days, frequency, monetary]
build_extra_features(tx, cutoff) -> DataFrame[customer_id, ...]
build_targets(tx, cutoff, days=config.HOLDOUT_DAYS) -> DataFrame[customer_id, churned, next_period_spend]
build_customer_table(tx, cutoff) -> one row per customer, features + targets
```
Sanity checks to print: number of customers per table, churn rate per table, `max(invoice_date)` used for features must be `< cutoff` (assert it).

---

## 9. Module 1: Segmentation (notebook 03)

- Inputs: `recency_days, frequency, monetary`, then `log1p` and `StandardScaler`, fit on train.
- **K-means**: k = 2..10, elbow (inertia) + silhouette + Davies–Bouldin to choose k (expect 3–5).
- **Hierarchical**: Ward linkage + dendrogram (fine for ~4–5k customers; subsample if slow).
- **GMM**: number of components chosen by BIC; compare with K-means.
- **PCA** to 2D for a scatter plot coloured by cluster, plus explained variance.
- **t-SNE for visualisation ONLY** (perplexity ~30), never as clustering input. An examiner may ask about this.
- Profile table: mean R, F, M and size per cluster. Give the clusters business names, e.g. *Champions, Loyal, At-risk, Lost / One-time*.
- Fit on train, `predict` on test; add `segment` to both tables and save.
- Metrics: silhouette, Davies–Bouldin, plus interpretability.

## 10. Module 2: Churn prediction (notebook 04)

- Target `churned`; features from Section 8 + one-hot `segment`.
- **Baselines:** (a) majority class, (b) a simple rule, `recency_days > 90 → churn`. Every ML model must beat these.
- Models: Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, SVM (SVM and LR need scaling, so use a `Pipeline`).
- Imbalance: `class_weight="balanced"`, or XGBoost `scale_pos_weight`.
- Tuning: `GridSearchCV` / `RandomizedSearchCV` with `StratifiedKFold(5)` on the **train table only**; scoring = ROC-AUC.
- Evaluate on the test table: ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix, ROC and PR curves.
- Threshold choice discussed (business trade-off: missing a churner vs wasting an offer).
- Feature importance (RF/XGB) and LR coefficients; short error analysis.
- Save results to `reports/results/churn_results.csv`.
- Planned depth: tune **Logistic Regression** and **XGBoost (or RF)** properly; the others stay as untuned comparison rows.

## 11. Module 3: Spend / CLV prediction (notebook 05)

- Target `next_period_spend`, modelled as `log1p(spend)` (right-skewed with many zeros); predictions converted back with `expm1`.
- **Baseline:** predict `revenue_last_90d` (spend in the previous 90 days) as the next 90 days' spend.
- Models: Linear Regression, Ridge, Lasso (scaled, alpha tuned with CV), Gradient Boosting Regressor (optionally XGBRegressor).
- Metrics on the test table, on the original £ scale: RMSE, MAE, R², plus R² on the log scale.
- Discuss Lasso feature selection (zeroed coefficients) and GBR feature importance.
- Optional: a two-part model (P(buy) from Module 2 × spend-given-buy).
- **Integration:** `retention_priority = P(churn) × predicted_spend`. Show a churn-risk vs predicted-spend quadrant (Retain now / Reward loyalty / Low-cost win-back / Nurture).
- Save results to `reports/results/clv_results.csv`.

## 12. (Optional) Module 4: Recommender (notebook 06)

Only if Modules 1–3 are finished by the end of week 4.
- Implicit feedback: customer × product purchase matrix (no ratings exist in this data, so say so explicitly).
- Item-based collaborative filtering (cosine similarity); optionally ALS.
- Evaluate with precision@k / recall@k on products bought in the test label window.
- Known issue: cold start for new products and customers.

---

## 13. Paper / report plan

**Sections:** Title, Abstract, Introduction, Related Work, Dataset & Preprocessing, Methodology (time split + 3 modules), Experiments & Results, Discussion (integration + business use), Limitations, Conclusion, References.

**Planned intro figures** (conceptual, no fake data):
1. **Figure 1: System overview.** Transaction data + cleaning, then the time-based split, then the customer table (X features / y targets), then the three modules. Churn is on top, segmentation in the middle, CLV at the bottom, and segment-label arrows run from Module 1 to Modules 2 and 3. All three feed a "Decisions" box. *Drafted (matplotlib).*
2. **Figure 2: Leakage-free time split.** A timeline from Dec 2009 to Dec 2011 with observation and holdout windows and the cutoff line. Three illustrative customers (regular, lapsed, new) show how R, F, M come from before the cutoff and the churn/spend labels from after. Optionally, a strip showing a random split mixing past and future.
3. **Figure 3: Decision framework.** A 2×2 quadrant of churn probability (Module 2) vs predicted spend (Module 3): *Retain now* (high risk, high value), *Reward loyalty*, *Low-cost win-back*, *Nurture / upsell*. Segment (Module 1) tailors the message.

**Figure style rules:**
- Same colour per module everywhere: Segmentation `#2a78d6` (blue), Churn `#eb6834` (orange), CLV `#1baf7a` (green). Neutral grey for everything else.
- Sans font (Liberation Sans / Arial), 6–8 pt at final size, PNG at 300 dpi + PDF.
- Text in dark ink, not in the series colour.
- Save into `reports/figures/` as `figXX_short_name.png`.

**Results figures to produce:** cleaning row-count table, EDA plots, elbow/silhouette, PCA and t-SNE cluster plots, segment profile table, ROC and PR curves (all models), confusion matrix (best model), feature importance, CLV predicted-vs-actual scatter, model comparison tables per module.

**References used in the proposal (IEEE style):**
1. D. Chen, S. L. Sain and K. Guo, "Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining," *J. Database Marketing & Customer Strategy Management*, vol. 19, pp. 197–208, 2012.
2. D. Chen, "Online Retail II," UCI Machine Learning Repository, 2019. https://doi.org/10.24432/C5CG6D
3. W. Verbeke, K. Dejaeger, D. Martens, J. Hur and B. Baesens, "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach," *European J. Operational Research*, vol. 218, no. 1, pp. 211–229, 2012.
4. P. S. Fader, B. G. S. Hardie and K. L. Lee, "RFM and CLV: Using iso-value curves for customer base analysis," *J. Marketing Research*, vol. 42, no. 4, pp. 415–430, 2005.
5. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
(If the recommender is added back: Koren, Bell & Volinsky 2009 "Matrix factorization techniques for recommender systems"; Hu, Koren & Volinsky 2008 "Collaborative filtering for implicit feedback datasets".)

---

## 14. Timeline (5 weeks) and work split

| Week | Work | Owner (proposed, change if needed) |
|---|---|---|
| 1 | Download data, `cleaning.py` + 01, `features.py` + 02 | **Both together** (everything depends on this) |
| 2 | Module 1 Segmentation (03) | Muhammad Hassam |
| 3 | Module 2 Churn (04) | Muhammad Hassam |
| 4 | Module 3 CLV (05) + integration quadrant; recommender only if ahead | Muhammad Salman Saeed |
| 5 | Final report, figures, slides, buffer | Both |

Salman also owns `src/` maintenance and `evaluation.py`.

---

## 15. Coding conventions (for us and for Claude Code)

- Windows machine; the project is at `D:\customer-intelligence\`. Use `pathlib` paths from `config.py` and never hard-code `D:\...`.
- **Python environment:** always use the project venv, `.venv\Scripts\python.exe` (Python 3.14, packages from `requirements.txt`). Do **not** use the bare `python` on PATH: on Salman's PC that is the MSYS2 Python (`C:\msys64\ucrt64\bin\python.exe`), which has none of the packages. Start Jupyter with `.venv\Scripts\python -m jupyter notebook`.
- **Never type a date or file path inside a notebook. Import it from `config`.**
- Reusable logic goes in `src/`; notebooks stay thin (call functions, show tables and plots, write short markdown explanations).
- Use sklearn `Pipeline` for scaling + model, so scaling never sees test data.
- `random_state=config.RANDOM_STATE` on every model and split.
- Save every figure with `plt.savefig(config.FIGURES_DIR / "name.png", dpi=300, bbox_inches="tight")`.
- Save every metric table as CSV in `config.RESULTS_DIR`.
- Keep code simple and well commented. **We are students and must be able to explain every line in the viva.** Prefer clear code over clever code, and explain new concepts briefly when introducing them.
- Before adding a new library, ask (keep `requirements.txt` small).
- After finishing a step, update the status checklist in Section 3.

---

## 16. Viva / interview questions to be ready for

- **Why a time-based split?** It prevents leakage: features are from the past only and labels from the future only, which mirrors real deployment.
- **Why two cutoffs?** Train and test come from different time periods, so the test score reflects true future performance.
- **How is churn defined?** No purchase in the 90 days after the cutoff, for customers who bought at least once before it. It is a business definition, because there is no "cancel subscription" event in retail.
- **Why log-transform spend?** It is heavily right-skewed with many zeros, and `log1p` stabilises variance for linear models.
- **Why was t-SNE not used for clustering?** It distorts global distances and densities. It is only for visualisation.
- **Class imbalance?** Class weights / `scale_pos_weight`, judged with PR-AUC and F1 rather than accuracy.
- **Why Ridge/Lasso?** Regularisation handles correlated RFM features, and Lasso also performs feature selection.
- **Limitations:** no demographic data, many wholesalers (very large orders), a cutoff date choice that affects labels, a single retailer (generalisation), and for the recommender, implicit feedback and cold start.
- **How do the modules connect?** Segment labels feed churn and CLV, and churn × CLV gives the retention priority.
