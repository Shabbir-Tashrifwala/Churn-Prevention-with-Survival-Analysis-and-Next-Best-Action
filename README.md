Please visit the project page: https://www.tashrifwala.com/projects/churn-prevention.html


# Telco Churn Prevention with Survival Analysis → Next-Best-Action (NBA)

End-to-end project that:
- models time-to-churn with Survival Analysis (CoxPH + GBM),
- translates predictions into prescriptions via a pragmatic Next-Best-Action (NBA) matrix (Risk × Value),
- and operationalizes the solution for both batch campaigns and a real-time FastAPI API.


Notebook: `churn_survival_nba.ipynb` 

---

## Table of Contents
- [Highlights](#highlights)
- [Repo Structure](#repo-structure)
- [Quickstart](#quickstart)
- [Batch Scoring (Path A)](#batch-scoring-path-a)
- [Real-Time API (Path B)](#real-time-api-path-b)
- [Data and Ethics](#data-and-ethics)
- [Reproducibility Notes](#reproducibility-notes)

---

## Highlights
- Exploratory survival: Kaplan–Meier curves and log-rank tests (Contract, InternetService, PaymentMethod, etc.).
- Pragmatic Feature Engineering.
- Modeling:
  - CoxPH for interpretable hazard ratios.
  - GBM (scikit-survival) for stronger predictive performance.
- Evaluation: C-index (ranking), time-dependent Brier (calibration), and Lift@H for business impact.
- NBA: Risk (by S(H)) × Value (CLV or charges quantiles) to Next-Best-Action, with a per-segment 10% control holdout for future uplift/ROI measurement.
- Operationalization:
  - Batch: `score_batch.py` generates campaign files at 1/3/6 months.
  - API: `app/main.py` serves `/predict` with FastAPI.
- Ready model included: `models/survival_pipeline.joblib` (<1 MB) for immediate runs.

---

## Repo Structure
```
├─ churn_survival_nba.ipynb           # One notebook: Sections 1–7
├─ score_batch.py                     # Batch scoring (Path A)
├─ app/
│  └─ main.py                         # FastAPI app (Path B)
├─ models/
│  └─ survival_pipeline.joblib        # Serialized pipeline (prep + model)
├─ data/
│  ├─ telco_sample_50rows.csv         # Sample for quick batch demo
│  ├─ api_example_one.json            # Single-row API payload
│  └─ api_example_payloads.json       # 3-row API payloads
├─ outputs/
│  ├─ section3/                       # Selected KM/log-rank plots and CSVs
│  └─ section6/                       # NBA segment/action tables and figures
├─ docs/
│  └─ Project Report.pdf              # Full write-up
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## Quickstart

Requires Python 3.10 or newer.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Verify the ready-to-run model exists
ls -lh models/survival_pipeline.joblib
```

---

## Batch Scoring (Path A)

Scores a CSV of active customers, computes survival at 1/3/6 months, derives `ChurnRiskScore = 1 − S(6m)`, segments by Risk × Value, assigns NextBestAction, and tags a small control holdout per segment.

```bash
python score_batch.py   --model-path models/survival_pipeline.joblib   --input-csv data/telco_sample_50rows.csv   --output-csv outputs/batch_scored.csv   --horizons 1 3 6   --risk-bands 0.70 0.90   --holdout-rate 0.10   --id-col customerID
```

Example output columns:
```
customerID, survival_prob_1m, survival_prob_3m, survival_prob_6m,
ChurnRiskScore, RiskSegment, ValueMetric, ValueSegment, NextBestAction, HoldoutControl, tenure
```

---

## Real-Time API (Path B)

Start the API (loads the same serialized pipeline once at startup):

```bash
uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs
```

Test with a single row:

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d @data/api_example_one.json
```

Example response:
```json
{
  "customer_id": "0001-ABCD",
  "survival_prob_1m": 0.98,
  "survival_prob_3m": 0.95,
  "survival_prob_6m": 0.90,
  "churn_risk_score": 0.10,
  "risk_segment": "Low",
  "value_segment": "High",
  "next_best_action": "Monitor & Nurture (exclusive content, loyalty rewards)"
}
```

---

## Data and Ethics
- Uses the public IBM Telco Customer Churn schema conventions for demonstration.
- Repository includes only a small sample CSV for quick runs (`data/telco_sample_50rows.csv`).
- No personal identifiers; outputs are educational and illustrative.

---

## Reproducibility Notes
- Single-notebook workflow (`churn_survival_nba.ipynb`).
- Pipelines are built with scikit-learn transformers so the same preprocessing runs in production.
- Serialized model (`models/survival_pipeline.joblib`) enables immediate execution without retraining.
- For consistent Value tiers in the API, you can freeze quantile cut points (v_lo, v_hi) into a small config and load on startup.
