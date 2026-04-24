# ICU Length of Stay Prediction with Conformal Quantile Regression
### MIMIC-IV · LightGBM · SHAP · Conformal Prediction

A three-stage clinical machine learning pipeline that predicts **calibrated ICU Length of Stay intervals** from the MIMIC-IV database. Rather than producing fragile point estimates, the system outputs rigorous prediction intervals with guaranteed marginal coverage — built for bed management and resource allocation in real clinical settings.

---

## Motivation

Point predictions for ICU Length of Stay are statistically fragile and practically insufficient for hospital operations. A model that outputs "5 days" provides no actionable uncertainty for bed planners. This project addresses that gap by combining:

- **Gradient Boosting with Quantile Pinball Loss** to learn the conditional distribution of LOS, not just its mean
- **SHAP explanations** to ensure every prediction is traceable to specific early physiological markers
- **Conformal Quantile Regression** to mathematically guarantee that the output intervals contain the true LOS with at least 90% marginal probability

---

## Repository Structure

```
.
├── 01_preprocessing.py       # Cohort construction, feature engineering, MICE imputation
├── 02_analysis.py            # Quantile LightGBM training, CV, SHAP analysis
├── 03_conformal.py           # Conformal calibration, interval construction, evaluation
└── README.md
```

---

## Pipeline Overview

```
MIMIC-IV Tables
      │
      ▼
01_preprocessing.py
  ├── ICU LOS target (icustays)
  ├── Demographics (patients + admissions, anchor-year corrected)
  ├── Baseline physiology (omr: height, weight, BMI, eGFR)
  ├── Lab features — first 24h (labevents)
  ├── Vital features — first 24h (chartevents)
  ├── Fluid input features — first 24h (inputevents)
  └── MICE imputation → cohort_preprocessed.parquet
      │
      ▼
02_analysis.py
  ├── 70/15/15 train/calibration/test split
  ├── LightGBM × 3 quantile models (q=0.1, 0.5, 0.9)
  ├── 5-fold cross-validation with Pinball Loss
  ├── SHAP TreeExplainer on median model
  └── Saves models (.pkl) + calibration/test sets (.parquet)
      │
      ▼
03_conformal.py
  ├── Nonconformity scores on calibration set
  ├── q_hat via finite-sample corrected quantile
  ├── Interval expansion on test set
  ├── Coverage evaluation (global + conditional by LOS stratum)
  ├── Winkler Score efficiency metric
  └── conformal_predictions.parquet + plots
```

---

## Data Source

This project uses **MIMIC-IV v3.1**, a large deidentified dataset of patients admitted to the ICU or emergency department at Beth Israel Deaconess Medical Center, Boston. Access requires credentialing through [PhysioNet](https://physionet.org/content/mimiciv/).

MIMIC-IV contains data for over **65,000 ICU patients** across **94,458 unique ICU stays**. The database is organized into two modules:

| Module | Key Tables Used |
|--------|----------------|
| `hosp` | `patients`, `admissions`, `labevents`, `omr` |
| `icu`  | `icustays`, `chartevents`, `inputevents` |

**Privacy note:** All dates in MIMIC-IV are shifted into a future time window (2100–2200) independently per patient. The pipeline corrects for this using `anchor_year` and `anchor_year_group` columns to recover chronologically consistent patient ages and admission timelines.

---

## Key Design Decisions

### Target Variable
ICU LOS is derived from the `icustays` table as `outtime − intime` in days. Each `stay_id` is treated as an independent prediction instance, including ICU readmissions within the same hospitalization, to accurately capture distinct physiological trajectories.

### Feature Extraction Window
All time-series features (labs, vitals, fluid inputs) are restricted to the **first 24 hours of ICU admission**. This is a strict leakage prevention boundary — no future data enters the model.

### Missing Data
Clinical data is Missing Not At Random. Simple mean imputation is explicitly avoided. **Multiple Imputation by Chained Equations (MICE)** is applied via `IterativeImputer` with 10 iterations to preserve the covariance structure of missing physiological measurements.

### Model Objective
Three separate LightGBM models are trained with `objective="quantile"` at α ∈ {0.1, 0.5, 0.9}. Tree-based models are chosen because they natively handle high-cardinality categorical variables from administrative data without sparse matrix expansion.

### Conformal Calibration
Raw quantile outputs are calibrated using **Conformal Quantile Regression** on a held-out calibration set (≈15% of data). The conformal correction `q_hat` is computed as:

```
q_hat = Quantile(scores, ⌈(n + 1)(1 − α)⌉ / n)
```

where nonconformity scores are `max(q_low − y, y − q_upper)` for each calibration point. This provides a **finite-sample marginal coverage guarantee** without any distributional assumptions.

---

## Outputs

| File | Description |
|------|-------------|
| `cohort_preprocessed.parquet` | Final feature matrix with MICE-imputed values |
| `lgbm_q10.pkl` | Trained lower quantile model (q=0.1) |
| `lgbm_q50.pkl` | Trained median model (q=0.5) |
| `lgbm_q90.pkl` | Trained upper quantile model (q=0.9) |
| `shap_importance_q50.csv` | Feature ranking by mean absolute SHAP value |
| `shap_summary_q50.png` | SHAP beeswarm summary plot |
| `conformal_predictions.parquet` | Test set predictions with bounds and per-patient coverage flags |
| `conformal_metrics.csv` | Aggregate coverage and efficiency metrics |
| `conformal_intervals.png` | Prediction interval plot sorted by observed LOS |
| `calibration_scores.png` | Histogram of calibration nonconformity scores |

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Pinball Loss** | Primary training and validation objective per quantile |
| **Empirical Coverage** | Proportion of test patients whose true LOS falls within the interval; target ≥ 90% |
| **Mean / Median Interval Width** | Efficiency of the prediction interval in days |
| **Conditional Coverage** | Coverage stratified by LOS stratum (<1d, 1–3d, 3–7d, 7–14d, >14d) |
| **Winkler Score** | Joint measure of interval width and coverage penalty |
| **MAE (median model)** | Point prediction accuracy of the q=0.5 model |

---

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib joblib pyarrow
```

Python 3.9+ is recommended. All dependencies are standard; no custom packages are required.

---

## Usage

Set the path to your local MIMIC-IV installation in `01_preprocessing.py`:

```python
MIMIC_PATH = "/path/to/mimic-iv/"
```

Then run the three scripts in order:

```bash
python 01_preprocessing.py
python 02_analysis.py
python 03_conformal.py
```

Each script reads the outputs of the previous stage from the working directory. For single-patient inference at deployment time, use the `predict_single_patient()` function in `03_conformal.py` directly.

---

## Data Access

MIMIC-IV is available on PhysioNet under a data use agreement. Access requires:

1. Completion of a recognized human research ethics training course (e.g. CITI Program)
2. Signing the PhysioNet Credentialed Health Data Use Agreement
3. Submitting a data access request at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)

---

## Citation

If you use MIMIC-IV in your work, please cite:

> Johnson, A., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T., Hao, S., Moody, B., Gow, B., Lehman, L., Celi, L. A., & Mark, R. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. https://doi.org/10.1038/s41597-022-01899-x

---

## License

This repository contains only analytical code. The MIMIC-IV data itself is governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/). Redistribution of the data in any form is prohibited.
