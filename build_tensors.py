"""
frames_long + passes -> padded tensors for the set/graph models, plus the optimizer's candidates.

Outputs (data/tensors/):
    passes_model.parquet   cleaned pass table with folds; row i <-> tensor row i
    tensors.npz            P [n,22,10] ppos [n,22,2] pteam [n,22] pmask [n,22]
                           Q [n,9] qpos [n,2] C [n,f_ctx] y_success y_shot10
    candidates.parquet     one row per (pass, candidate target): every visible teammate
                           (not the actor or keeper) plus the actual target; holds the raw
                           ACTION features so both tabular and deep models can score it
    meta.json              feature names, context categories, sizes

Usage:
    python build_tensors.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import (ACTION, N_MAX, PLAYER_FEATS, QUERY_FEATS, TENSORS, action_features, angle_to_goal,
                    clean_passes, load_raw, query_features, write_json)

CTX_CATS = ["play_pattern", "pass_type"]


def context_features(p: pd.DataFrame, cats: dict) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for c in CTX_CATS:
        for v in cats[c]:
            cols.append((p[c] == v).values.astype(np.float32)); names.append(f"{c}={v}")
    for name, v in [("under_pressure", p.under_pressure), ("score_diff/3", p.score_diff / 3),
                    ("minute/90", p.minute / 90), ("visible_area/8000", p.visible_area / 8000),
                    ("n_visible_teammates/11", p.n_visible_teammates / 11),
                    ("n_visible_opponents/11", p.n_visible_opponents / 11)]:
        cols.append(np.asarray(v, np.float32)); names.append(name)
    return np.stack(cols, 1), names


def player_tensors(p: pd.DataFrame, frames: pd.DataFrame):
    """Vectorised scatter of frames_long into [n, N_MAX, ...] arrays, nearest-to-ball first."""
    row = pd.Series(np.arange(len(p)), index=p.pass_id)
    f = frames[frames.pass_id.isin(row.index)].copy()
    f["row"] = row.loc[f.pass_id].values
    bx, by = p.ball_x.values[f.row.values], p.ball_y.values[f.row.values]
    f["dx"], f["dy"] = f.x.values - bx, f.y.values - by
    f["db"] = np.hypot(f.dx, f.dy)
    f = f.sort_values(["row", "db"], kind="stable")
    f["rank"] = f.groupby("row").cumcount()
    trunc = int((f["rank"] >= N_MAX).sum())
    f = f[f["rank"] < N_MAX]

    n = len(p)
    xy = f[["x", "y"]].values
    feats = np.stack([
        f.x / 120, f.y / 80, f.teammate.astype(float), f.actor.astype(float), f.keeper.astype(float),
        f.dx / 50, f.dy / 50, f.db / 50, np.linalg.norm(xy - [120.0, 40.0], axis=1) / 120, angle_to_goal(xy),
    ], 1).astype(np.float32)
    r, k = f.row.values, f["rank"].values
    P = np.zeros((n, N_MAX, len(PLAYER_FEATS)), np.float32); P[r, k] = feats
    ppos = np.zeros((n, N_MAX, 2), np.float32); ppos[r, k] = xy
    pteam = np.zeros((n, N_MAX), np.int8); pteam[r, k] = f.teammate.values
    pmask = np.zeros((n, N_MAX), bool); pmask[r, k] = True
    return P, ppos, pteam, pmask, trunc


def candidates(p: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """For every pass: each visible teammate (not actor / keeper) as a hypothetical target + the actual target."""
    row = pd.Series(np.arange(len(p)), index=p.pass_id)
    f = frames[frames.pass_id.isin(row.index)].copy()
    f["row"] = row.loc[f.pass_id].values
    f = f.sort_values(["row", "slot"])
    rows_arr, xy = f.row.values, f[["x", "y"]].values
    tm, act, gk = f.teammate.values, f.actor.values, f.keeper.values
    bounds = np.flatnonzero(np.diff(rows_arr)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(f)]
    ball = p[["ball_x", "ball_y"]].values
    end_actual = p[["end_x", "end_y"]].values

    out = []
    for s, e in zip(starts, ends):
        i = rows_arr[s]
        opp = xy[s:e][~tm[s:e]]
        cmask = tm[s:e] & ~act[s:e] & ~gk[s:e]
        targets = np.vstack([end_actual[i:i + 1], xy[s:e][cmask]])
        a = action_features(ball[i], targets, opp)
        a["row"] = np.full(len(targets), i)
        a["slot"] = np.r_[-1, f.slot.values[s:e][cmask]]
        a["is_actual"] = np.r_[1, np.zeros(cmask.sum(), int)]
        out.append(pd.DataFrame(a))
    c = pd.concat(out, ignore_index=True)
    c.insert(0, "pass_id", p.pass_id.values[c.row.values])
    return c


def main():
    t0 = time.time()
    passes, frames = load_raw()
    p = clean_passes(passes, frames)
    print(f"cleaned: {len(passes):,} -> {len(p):,} passes "
          f"(train {int((p.split == 'train').sum()):,}, test {int((p.split == 'test').sum()):,})")

    cats = {c: sorted(p[c].unique().tolist()) for c in CTX_CATS}
    C, ctx_names = context_features(p, cats)
    P, ppos, pteam, pmask, trunc = player_tensors(p, frames)
    Q = query_features(p)
    qpos = p[["end_x", "end_y"]].values.astype(np.float32)
    print(f"players: mean visible {pmask.sum(1).mean():.1f}, truncated rows {trunc}")

    TENSORS.mkdir(parents=True, exist_ok=True)
    p.to_parquet(TENSORS / "passes_model.parquet", index=False)
    np.savez(TENSORS / "tensors.npz", P=P, ppos=ppos, pteam=pteam, pmask=pmask, Q=Q, qpos=qpos, C=C,
             y_success=p.y_success.values.astype(np.float32), y_shot10=p.y_shot10.values.astype(np.float32))

    cand = candidates(p, frames)
    # sanity check: recomputed features of the actual target must reproduce clean_pipeline.py
    act = cand[cand.is_actual == 1].set_index("row").sort_index()
    for col in ACTION:
        err = np.abs(act[col].values - p[col].values).max()
        assert err < 1e-6, f"candidate feature {col} disagrees with passes.parquet (max err {err})"
    cand.to_parquet(TENSORS / "candidates.parquet", index=False)
    n_alt = int((cand.is_actual == 0).sum())
    print(f"candidates: {len(cand):,} rows ({n_alt / len(p):.1f} teammates per pass); actual-target check OK")

    write_json({"n": len(p), "N_MAX": N_MAX, "player_feats": PLAYER_FEATS, "query_feats": QUERY_FEATS,
                "ctx_feats": ctx_names, "ctx_cats": cats, "f_player": P.shape[-1], "f_query": Q.shape[-1],
                "f_ctx": C.shape[-1], "truncated_player_rows": trunc}, TENSORS / "meta.json")
    print(f"done in {time.time() - t0:.0f}s -> {TENSORS}")


if __name__ == "__main__":
    main()
