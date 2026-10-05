"""
Post-hoc calibration fitted on out-of-fold (train-split) predictions only, then applied to test.

    trees (B0b, B2*, B3*)          -> isotonic regression
    neural / linear (B1*, M*, deep) -> temperature scaling  p = sigmoid(logit(p) / T)
    B0 prior                       -> none

Reads results/preds/*.parquet, adds a `p_cal` column and writes results/calibrated/*.parquet;
calibrators go to results/models/calibrators/ (the optimizer reuses them).

Usage:
    python calibrate.py
"""
from __future__ import annotations

import joblib
import pandas as pd

from calibrators import method_for
from common import MODELS, PREDS, RESULTS

OUT = RESULTS / "calibrated"


def calibrate_file(path):
    name, task = path.stem.split("__")
    df = pd.read_parquet(path)
    fit_rows = (df.split == "train") & df.p.notna()
    cal = method_for(name).fit(df.p[fit_rows].values, df.y[fit_rows].values)
    df["p_cal"] = cal(df.p.values)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / path.name, index=False)
    (MODELS / "calibrators").mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, MODELS / "calibrators" / f"{name}__{task}.joblib")
    return name, task, type(cal).__name__, getattr(cal, "T", None), int(fit_rows.sum())


def main():
    for path in sorted(PREDS.glob("*__*.parquet")):
        name, task, kind, T, n = calibrate_file(path)
        print(f"{name:20s} {task:8s} {kind:12s}" + (f" T={T:.3f}" if T else "") + f"  (fit on {n:,} OOF rows)")


if __name__ == "__main__":
    main()
