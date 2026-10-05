"""Post-hoc calibrators (importable so pickled calibrators load from any script)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from common import logit, sigmoid


class Identity:
    def fit(self, p, y): return self
    def __call__(self, p): return np.asarray(p, float)


class Temperature:
    def fit(self, p, y):
        z, y = logit(p), np.asarray(y, float)

        def nll(log_t):
            q = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def __call__(self, p):
        return sigmoid(logit(p) / self.T)


class Isotonic:
    def fit(self, p, y):
        self.m = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p, y); return self

    def __call__(self, p):
        return self.m.predict(np.asarray(p, float))


def method_for(name: str):
    if name == "B0":
        return Identity()
    if name.startswith(("B0b", "B2", "B3")):
        return Isotonic()
    return Temperature()
