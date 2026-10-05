"""
Shared definitions for the modelling stage: paths, feature sets, geometry,
cleaning (design doc of the data pipeline, Section 7), folds and metrics.

Every script imports from here so that the tabular track, the tensor track and
the optimizer all agree on what a "decision-time" feature is.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------------- paths
DATA = Path("data")
PROC = DATA / "processed"
TENSORS = DATA / "tensors"
RESULTS = Path("results")
PREDS = RESULTS / "preds"
MODELS = RESULTS / "models"
FIGS = RESULTS / "figures"

N_FOLDS = 5
N_MAX = 22
GOAL = np.array([120.0, 40.0])
POST_L, POST_R = np.array([120.0, 36.0]), np.array([120.0, 44.0])

# ----------------------------------------------------------------------------- feature sets
STATE = ["under_pressure", "score_diff", "period", "minute", "ball_x", "ball_y", "ball_dist_goal",
         "ball_angle_goal", "opp_within_5", "nearest_opp_dist", "opp_in_cone", "def_line_x", "tm_ahead",
         "n_visible_teammates", "n_visible_opponents", "visible_area"]
ACTION = ["end_x", "end_y", "pass_len", "pass_angle", "progress_x", "end_dist_goal", "end_angle_goal",
          "opp_near_target_3", "lane_opp_2", "target_in_box"]
LOCATION = ["ball_x", "ball_y", "end_x", "end_y"]
# Decision-time (DT) categoricals: knowable before the pass for ANY candidate target.
CAT_DT = ["play_pattern", "pass_type"]
# FULL adds things observed only after the pass is played (prediction only, never the optimizer).
CAT_FULL = CAT_DT + ["body_part", "pass_height"]
FEATURE_SETS = {"DT": (STATE + ACTION, CAT_DT), "FULL": (STATE + ACTION, CAT_FULL), "LOC": (LOCATION, [])}

TASKS = {  # task -> (label column, restrict to completed passes?)
    "success": ("y_success", False),
    "shot10": ("y_shot10", True),
}


# ----------------------------------------------------------------------------- geometry
def angle_to_goal(p: np.ndarray) -> np.ndarray:
    """Opening angle (radians) between the posts, vectorised over p [...,2]."""
    a, b = POST_L - p, POST_R - p
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    t = np.clip(((pts - a) @ v) / (v @ v + 1e-9), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * v), axis=1)


def action_features(ball: np.ndarray, ends: np.ndarray, opp: np.ndarray) -> dict:
    """The 10 target-dependent features, identical to clean_pipeline.py, for M candidate targets.

    ball [2], ends [M,2], opp [K,2] (visible opponents)  ->  dict of [M] arrays
    """
    d = ends - ball
    if len(opp):
        near = (np.linalg.norm(opp[None] - ends[:, None], axis=-1) < 3).sum(1)
        t = np.clip(((opp[None] - ball) * d[:, None]).sum(-1) / ((d * d).sum(-1)[:, None] + 1e-9), 0, 1)
        lane = (np.linalg.norm(opp[None] - (ball + t[..., None] * d[:, None]), axis=-1) < 2).sum(1)
    else:
        near = lane = np.zeros(len(ends), int)
    return {
        "end_x": ends[:, 0], "end_y": ends[:, 1],
        "pass_len": np.linalg.norm(d, axis=1),
        "pass_angle": np.arctan2(d[:, 1], d[:, 0]),
        "progress_x": d[:, 0],
        "end_dist_goal": np.linalg.norm(GOAL - ends, axis=1),
        "end_angle_goal": angle_to_goal(ends),
        "opp_near_target_3": near,
        "lane_opp_2": lane,
        "target_in_box": ((ends[:, 0] >= 102) & (ends[:, 1] >= 18) & (ends[:, 1] <= 62)).astype(int),
    }


# ----------------------------------------------------------------------------- token features
PLAYER_FEATS = ["x/120", "y/80", "teammate", "actor", "keeper", "dx/50", "dy/50", "dist_ball/50",
                "dist_goal/120", "angle_goal"]
QUERY_FEATS = ["end_x/120", "end_y/80", "pass_len/50", "progress_x/50", "end_dist_goal/120",
               "end_angle_goal", "opp_near_target_3", "lane_opp_2", "target_in_box"]
# indices whose value changes under the vertical mirror y -> 80 - y
PLAYER_Y_FLIP, PLAYER_Y_NEG = [1], [6]
QUERY_Y_FLIP = [1]


def query_features(a) -> np.ndarray:
    """Query-token features [M,9] from a DataFrame / dict holding the ACTION columns."""
    return np.stack([np.asarray(a["end_x"]) / 120, np.asarray(a["end_y"]) / 80, np.asarray(a["pass_len"]) / 50,
                     np.asarray(a["progress_x"]) / 50, np.asarray(a["end_dist_goal"]) / 120,
                     np.asarray(a["end_angle_goal"]), np.asarray(a["opp_near_target_3"]),
                     np.asarray(a["lane_opp_2"]), np.asarray(a["target_in_box"])], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- data
def load_raw():
    passes = pd.read_parquet(PROC / "passes.parquet")
    frames = pd.read_parquet(PROC / "frames_long.parquet")
    return passes, frames


def clean_passes(passes: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-training checklist and attach GroupKFold folds (train only; test fold = -1)."""
    actor = frames[frames.actor].drop_duplicates("pass_id").set_index("pass_id")[["x", "y"]]
    p = passes.join(actor, on="pass_id")
    mismatch = np.hypot(p.x - p.ball_x, p.y - p.ball_y) > 2.0
    keep = ~mismatch & (p.pass_len >= 0.5) & p.nearest_opp_dist.notna() & p.def_line_x.notna()
    p = p.loc[keep].drop(columns=["x", "y"]).reset_index(drop=True)
    p["visible_area"] = p.visible_area.fillna(p.loc[p.split == "train", "visible_area"].median())
    p["fold"] = -1
    tr = np.flatnonzero(p.split == "train")
    for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(tr, groups=p.match_id.values[tr])):
        p.loc[tr[va], "fold"] = k
    return p


def load_model_table() -> pd.DataFrame:
    """The cleaned, fold-annotated pass table written by build_tensors.py (row order == tensor order)."""
    return pd.read_parquet(TENSORS / "passes_model.parquet")


def design_matrix(p: pd.DataFrame, featset: str, columns: list[str] | None = None) -> pd.DataFrame:
    num, cat = FEATURE_SETS[featset]
    X = pd.get_dummies(p[num + cat], columns=cat, dtype=float)
    if columns is not None:  # align to the training columns (e.g. for candidates)
        X = X.reindex(columns=columns, fill_value=0.0)
    return X.astype(float)


def task_mask(p: pd.DataFrame, task: str) -> np.ndarray:
    _, completed_only = TASKS[task]
    return (p.y_success == 1).values if completed_only else np.ones(len(p), bool)


# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def metrics(y, p, prior: float | None = None) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"n": len(y), "logloss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
           "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
           "prauc": average_precision_score(y, p), "ece": ece(y, p)}
    if prior is not None:
        out["bss"] = 1 - out["brier"] / brier_score_loss(y, np.full_like(y, prior))
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def save_preds(df: pd.DataFrame, name: str, task: str):
    """Unified prediction file: pass_id, split, fold, y, p  (OOF on train, refit on test)."""
    PREDS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREDS / f"{name}__{task}.parquet", index=False)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
