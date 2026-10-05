"""
Tabular estimators (B0, B0b, B1, B2/B3, M1) and the Optuna tuner for XGBoost.

Kept in their own module so pickled models load from any script (train_tabular, optimize).
All estimators expose fit(X, y) and predict_proba(X) -> P(y = 1) as a 1-D array.
"""
from __future__ import annotations

import numpy as np
import optuna
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import N_FOLDS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


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
