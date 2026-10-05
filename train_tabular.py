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
import optuna
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (MODELS, N_FOLDS, TASKS, design_matrix, load_model_table, metrics, save_preds, task_mask,
                    write_json)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = MODELS / "tabular"


# ----------------------------------------------------------------------------- B0 / B0b
class Prior:
    def fit(self, X, y):
        self.p = float(np.mean(y)); return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


class GridLookup:
    """Empirical rate per (start zone x end zone) on a 12x8 grid, smoothed toward the prior."""

    def __init__(self, nx=12, ny=8, alpha=2.0):
        self.nx, self.ny, self.alpha = nx, ny, alpha

    def _cell(self, X):
        def z(x, y):
            return (np.clip((x / 120 * self.nx).astype(int), 0, self.nx - 1) * self.ny
                    + np.clip((y / 80 * self.ny).astype(int), 0, self.ny - 1))
        return z(X.ball_x.values, X.ball_y.values) * self.nx * self.ny + z(X.end_x.values, X.end_y.values)

    def fit(self, X, y):
        c, k = self._cell(X), (self.nx * self.ny) ** 2
        self.prior = float(np.mean(y))
        pos, n = np.bincount(c, weights=y, minlength=k), np.bincount(c, minlength=k)
        self.rate = (pos + self.alpha * self.prior) / (n + self.alpha)
        return self

    def predict_proba(self, X):
        return self.rate[self._cell(X)]


# ----------------------------------------------------------------------------- B1
class Logistic:
    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, y); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------- B2 / B3
XGB_BASE = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                device=DEV, n_jobs=4, verbosity=0)


class XGB:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.m = xgb.XGBClassifier(**XGB_BASE, **self.params).fit(X, y); return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


def tune_xgb(X, y, folds, trials: int, seed: int = 0) -> dict:
    """Optuna search with grouped CV; returns params incl. n_estimators = mean best iteration."""
    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 50, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 5, log=True),
        )
        losses, iters = [], []
        for k in range(N_FOLDS):
            tr, va = folds != k, folds == k
            m = xgb.XGBClassifier(**XGB_BASE, **params, n_estimators=2000, early_stopping_rounds=50,
                                  random_state=seed)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            losses.append(m.best_score); iters.append(m.best_iteration + 1)
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params, n_estimators=study.best_trial.user_attrs["n_estimators"])
    print(f"    optuna best CV logloss {study.best_value:.4f} with {best}", flush=True)
    return best


# ----------------------------------------------------------------------------- M1
class MLP:
    """3 x 128 ReLU MLP, dropout 0.2, early stopping on a 10% inner split of matches."""

    def __init__(self, groups=None, epochs=100, patience=5, lr=1e-3, bs=512, seed=0):
        self.groups, self.epochs, self.patience, self.lr, self.bs, self.seed = groups, epochs, patience, lr, bs, seed

    def _net(self, d):
        return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(128, 1)).to(DEV)

    def fit(self, X, y, groups=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        g = np.asarray(groups if groups is not None else np.arange(len(y)))
        ug = np.unique(g); val_g = rng.choice(ug, max(1, len(ug) // 10), replace=False)
        va = np.isin(g, val_g)
        self.sc = StandardScaler().fit(X[~va])
        Xt = torch.tensor(self.sc.transform(X), dtype=torch.float32, device=DEV)
        yt = torch.tensor(y, dtype=torch.float32, device=DEV)
        tr_idx = torch.tensor(np.flatnonzero(~va), device=DEV)
        va_idx = torch.tensor(np.flatnonzero(va), device=DEV)
        self.net = self._net(X.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = tr_idx[torch.randperm(len(tr_idx), device=DEV)]
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                opt.zero_grad(); lossf(self.net(Xt[b]).squeeze(-1), yt[b]).backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = lossf(self.net(Xt[va_idx]).squeeze(-1), yt[va_idx]).item()
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.cpu()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(self.sc.transform(X), dtype=torch.float32)).squeeze(-1)
        return torch.sigmoid(z).numpy()


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
