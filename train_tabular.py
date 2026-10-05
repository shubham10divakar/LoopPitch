"""
Tabular track: B0 prior, B0b grid lookup, B1 logistic, B2 XGBoost (Optuna-tuned), B3 XGBoost on
locations only, M1 MLP.

For each model and task it writes results/preds/{model}__{task}.parquet holding
    train rows: out-of-fold predictions (GroupKFold by match, fold column from build_tensors.py)
    test rows:  predictions of a model refit on the whole training split
and saves the refit model to results/models/tabular/ (the optimizer reuses the DT models).

Usage:
    python train_tabular.py                         # everything, 50 Optuna trials
    python train_tabular.py --models B0 B1 --tasks success
    python train_tabular.py --trials 10             # quicker tuning
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd

from common import (MODELS, N_FOLDS, TASKS, design_matrix, load_model_table, metrics, save_preds, task_mask,
                    write_json)
from tabular_models import MLP, XGB, GridLookup, Logistic, Prior, tune_xgb

OUT = MODELS / "tabular"


# ----------------------------------------------------------------------------- driver
MODEL_SPECS = {  # name -> (feature set, factory(task_context) -> estimator)
    "B0": ("DT", lambda ctx: Prior()),
    "B0b": ("GRID", lambda ctx: GridLookup()),
    "B1_DT": ("DT", lambda ctx: Logistic()),
    "B1_FULL": ("FULL", lambda ctx: Logistic()),
    "B2_DT": ("DT", lambda ctx: XGB(**ctx["xgb_params"]("DT"))),
    "B2_FULL": ("FULL", lambda ctx: XGB(**ctx["xgb_params"]("FULL"))),
    "B3_LOC": ("LOC", lambda ctx: XGB(**ctx["xgb_params"]("LOC"))),
    "M1_DT": ("DT", lambda ctx: MLP()),
}


def run_model(name, featset, factory, ctx, p, task):
    ycol, _ = TASKS[task]
    m = task_mask(p, task)
    sub = p.loc[m].reset_index(drop=True)
    y = sub[ycol].values.astype(float)
    if featset == "GRID":
        X = sub[["ball_x", "ball_y", "end_x", "end_y"]]
    else:
        X = design_matrix(sub, featset)
    tr = (sub.split == "train").values
    folds = sub.fold.values

    def fit(est, rows):
        if isinstance(est, MLP):
            return est.fit(X.values[rows], y[rows], groups=sub.match_id.values[rows])
        return est.fit(X[rows] if featset == "GRID" else X.values[rows], y[rows])

    def predict(est, rows):
        return est.predict_proba(X[rows] if featset == "GRID" else X.values[rows])

    pred = np.full(len(sub), np.nan)
    for k in range(N_FOLDS):
        trk, vak = tr & (folds != k), tr & (folds == k)
        pred[vak] = predict(fit(factory(ctx), trk), vak)
    final = fit(factory(ctx), tr)
    pred[~tr] = predict(final, ~tr)

    save_preds(pd.DataFrame({"pass_id": sub.pass_id, "split": sub.split, "fold": folds, "y": y, "p": pred}),
               name, task)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = None if featset == "GRID" else list(X.columns)
    joblib.dump({"model": final, "featset": featset, "columns": cols}, OUT / f"{name}__{task}.joblib")
    prior = y[tr].mean()
    return {"oof": metrics(y[tr], pred[tr], prior), "test": metrics(y[~tr], pred[~tr], prior)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--trials", type=int, default=50, help="Optuna trials per XGBoost model")
    args = ap.parse_args()

    p = load_model_table()
    summary = []
    for task in args.tasks:
        ycol, _ = TASKS[task]
        tuned = {}

        def xgb_params(featset, task=task):
            if featset not in tuned:
                sub = p.loc[task_mask(p, task) & (p.split == "train").values]
                X = design_matrix(sub, featset).values
                print(f"  tuning XGBoost [{task}/{featset}] with {args.trials} trials ...", flush=True)
                tuned[featset] = tune_xgb(X, sub[ycol].values, sub.fold.values, args.trials)
                write_json(tuned[featset], OUT / f"xgb_params__{task}__{featset}.json")
            return tuned[featset]

        ctx = {"xgb_params": xgb_params}
        for name in args.models:
            featset, factory = MODEL_SPECS[name]
            t0 = time.time()
            r = run_model(name, featset, factory, ctx, p, task)
            print(f"[{task}] {name:8s} OOF logloss {r['oof']['logloss']:.4f} | TEST logloss "
                  f"{r['test']['logloss']:.4f} auc {r['test']['auc']:.3f} prauc {r['test']['prauc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            for part in ("oof", "test"):
                summary.append({"task": task, "model": name, "part": part, **r[part]})
    df = pd.DataFrame(summary)
    path = MODELS.parent / "tabular_summary.csv"
    if path.exists():  # merge with earlier runs of other models / tasks
        old = pd.read_csv(path)
        old = old[~old.set_index(["task", "model"]).index.isin(df.set_index(["task", "model"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.round(4).to_csv(path, index=False)


if __name__ == "__main__":
    main()
