#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch scoring script for Telco churn survival model + NBA mapping.

Usage:
  python score_batch.py     --model-path models/survival_pipeline.joblib     --input-csv data/active_customers.csv     --output-csv outputs/batch_scored.csv     --horizons 1 3 6     --risk-bands 0.70 0.90     --value-col CLV     --holdout-rate 0.10

Notes:
- The model-path points to a single joblib artifact that contains the entire sklearn Pipeline
  (prep/transformers + survival model). This is produced by the notebook serialization cell.
- Input CSV must have the exact feature columns the pipeline expects.
- Outputs include survival probabilities at requested horizons, churn-risk at max horizon,
  risk/value segments, and NextBestAction per Section 6.
"""

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

# ---------- NBA config (mirrors Section 6) ----------
NBA_MATRIX = {
    ("Low",    "High"):   "Monitor & Nurture (exclusive content, loyalty rewards)",
    ("Medium", "High"):   "Proactive Check-in (CS call, early renewal bonus)",
    ("High",   "High"):   "High-Touch Intervention (Sr. AM outreach + significant offer)",

    ("Low",    "Medium"): "Standard Engagement (regular newsletter)",
    ("Medium", "Medium"): "Targeted Automated Campaign (highlight unused features)",
    ("High",   "Medium"): "Automated Retention Offer (email with ~15% renewal discount)",

    ("Low",    "Low"):    "Do Nothing (monitor only)",
    ("Medium", "Low"):    "Do Nothing (monitor only)",
    ("High",   "Low"):    "Low-Cost Nudge (in-app survey to gather feedback)",
}

def survival_at_h_from_stepfuncs(sf_list, h: float) -> np.ndarray:
    out = np.empty(len(sf_list), dtype=float)
    for i, sf in enumerate(sf_list):
        out[i] = float(sf(h))
    return out

def risk_tier_from_S(S_h: float, lo_med: float, hi_med: float) -> str:
    if S_h < lo_med:
        return "High"
    if S_h < hi_med:
        return "Medium"
    return "Low"

def pick_value_column(df_like: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df_like.columns:
        return preferred
    for col in ["CLV","clv","CustomerLifetimeValue","MonthlyCharges","TotalCharges"]:
        if col in df_like.columns:
            return col
    # Fallbacks
    if "MonthlyCharges" in df_like.columns:
        df_like["CLV_proxy"] = pd.to_numeric(df_like["MonthlyCharges"], errors="coerce") * 12.0
        return "CLV_proxy"
    df_like["ValueProxy"] = 1.0
    return "ValueProxy"

def compute_value_tiers(series: pd.Series, q_low=0.33, q_high=0.66) -> tuple[float,float]:
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanquantile(s, q_low)), float(np.nanquantile(s, q_high))

def value_tier(v: float, v_lo: float, v_hi: float) -> str:
    if pd.isna(v):
        return "Medium"
    if v < v_lo:
        return "Low"
    if v < v_hi:
        return "Medium"
    return "High"

def map_action(risk_seg: str, value_seg: str) -> str:
    return NBA_MATRIX.get((risk_seg, value_seg), "Do Nothing (monitor only)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--horizons", type=float, nargs="+", default=[1,3,6], help="Months")
    parser.add_argument("--risk-bands", type=float, nargs=2, default=[0.70, 0.90],
                        help="Survival S(h) cut points: [low/med, med/low] e.g., 0.70 0.90")
    parser.add_argument("--value-col", type=str, default=None, help="Preferred value column")
    parser.add_argument("--holdout-rate", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--id-col", type=str, default="customerID")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_csv  = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path} "
                                f"(serialize your pipeline first from the notebook)")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"[Batch] Loading model from {model_path} ...")
    pipe = joblib.load(model_path)
    if not hasattr(pipe, "named_steps"):
        warnings.warn("Loaded object is not a Pipeline; ensure you saved the full pipeline.")
    # Identify steps by names used in your notebook (prep, colfilter, model)
    steps = getattr(pipe, "named_steps", {})
    prep = steps.get("prep", None)
    colfilter = steps.get("colfilter", None)
    model = steps.get("model", None)

    print(f"[Batch] Reading input data {input_csv} ...")
    df = pd.read_csv(input_csv)

    # Preserve ID if present
    id_col = args.id_col if args.id_col in df.columns else None

    # Transform
    Xt = df
    if prep is not None:
        Xt = prep.transform(Xt)
    if colfilter is not None:
        Xt = colfilter.transform(Xt)

    if model is None and hasattr(pipe, "predict_survival_function"):
        # Some users may save just the estimator with prep baked in; try pipe itself.
        model_like = pipe
    else:
        model_like = model

    print(f"[Batch] Predicting survival at horizons: {args.horizons}")
    sf_list = model_like.predict_survival_function(Xt)
    # Build survival columns
    surv_cols = {}
    for h in args.horizons:
        surv_cols[f"survival_prob_{int(h)}m"] = survival_at_h_from_stepfuncs(sf_list, h)

    scored = df.copy()
    for k, v in surv_cols.items():
        scored[k] = v

    # Choose the max horizon as the operational risk
    H = max(args.horizons)
    S_h = surv_cols[f"survival_prob_{int(H)}m"]
    risk = 1.0 - S_h
    scored["ChurnRiskScore"] = risk

    # Risk segmentation based on S(H)
    lo_med, hi_med = args.risk_bands  # e.g., 0.70, 0.90
    scored["RiskSegment"] = [risk_tier_from_S(s, lo_med, hi_med) for s in S_h]

    # Value segmentation
    value_col = pick_value_column(scored, args.value_col)
    v_lo, v_hi = compute_value_tiers(scored[value_col], q_low=0.33, q_high=0.66)
    scored["ValueMetric"] = pd.to_numeric(scored[value_col], errors="coerce")
    scored["ValueSegment"] = [value_tier(v, v_lo, v_hi) for v in scored["ValueMetric"]]

    # NBA action
    scored["NextBestAction"] = [map_action(r, v) for r, v in zip(scored["RiskSegment"], scored["ValueSegment"])]

    # Control holdout per cell
    rng = np.random.default_rng(args.random_seed)
    holdout_flags = np.zeros(len(scored), dtype=bool)
    for (r, v), idx in scored.groupby(["RiskSegment","ValueSegment"]).indices.items():
        n = len(idx)
        k = int(np.floor(args.holdout_rate * n))
        if k > 0:
            pick = rng.choice(idx, size=k, replace=False)
            holdout_flags[pick] = True
    scored["HoldoutControl"] = holdout_flags

    # Order columns
    out_cols = []
    if id_col:
        out_cols.append(id_col)
    out_cols += list(surv_cols.keys()) + [
        "ChurnRiskScore", "RiskSegment", "ValueMetric", "ValueSegment", "NextBestAction", "HoldoutControl"
    ]
    # Include Tenure/tenure if available to help campaign teams
    for tcol in ["tenure","Tenure","tenure_months"]:
        if tcol in scored.columns:
            out_cols.append(tcol)

    # Save
    scored[out_cols].to_csv(output_csv, index=False)
    print(f"[Batch] Wrote {len(scored)} rows → {output_csv}")

if __name__ == "__main__":
    main()
