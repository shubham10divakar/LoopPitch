"""
Pass optimizer: score every visible teammate as a hypothetical target, EV = P_hat * Q_hat.

For every pass the candidate set is {actual target} U {visible teammates except actor and keeper}
(built by build_tensors.py). Decision gap: delta = EV(best candidate) - EV(actual) >= 0.

P and Q can come from a tabular DT model (B0b, B1_DT, B2_DT, B3_LOC, M1_DT; never FULL - those use
post-pass information) or from a deep run (results/deep/{name}/seed*_refit.pt, seed-ensembled).
Calibrators fitted by calibrate.py are applied when present.

Outputs (results/optimizer/{P}+{Q}/):
    candidates_scored.parquet, passes_scored.parquet, summary.json,
    by_team.csv, by_player.csv (>= --min-passes passes), decision_gap.png, case_*.png

Usage:
    python optimize.py --p-model B2_DT --q-model B1_DT
    python optimize.py --p-model M5_T4 --q-model M5_T4 --split all --case-player "Lamine Yamal"
    python optimize.py --p-model B2_DT --q-model B1_DT --plot-pass <pass_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import (ACTION, MODELS, PROC, RESULTS, TENSORS, design_matrix, load_model_table, logit,  # noqa: E402
                    query_features, sigmoid, write_json)
from looped_pitch_model import predict_probs  # noqa: E402
from train_deep import DEV, load_tensors, mirror, train_depth  # noqa: E402
from models_deep import build_model  # noqa: E402

TASK_COL = {"success": 0, "shot10": 1}


def calibrator(name, task):
    path = MODELS / "calibrators" / f"{name}__{task}.joblib"
    return joblib.load(path) if path.exists() else (lambda p: p)


def score_tabular(name, task, cand, table):
    obj = joblib.load(MODELS / "tabular" / f"{name}__{task}.joblib")
    if obj["featset"] == "FULL":
        raise ValueError(f"{name} uses post-pass features (FULL) and cannot score hypothetical targets")
    X = table.iloc[cand.row.values].reset_index(drop=True).drop(columns=ACTION)
    X = pd.concat([X, cand[ACTION].reset_index(drop=True)], axis=1)
    if obj["featset"] == "GRID":
        return obj["model"].predict_proba(X[["ball_x", "ball_y", "end_x", "end_y"]])
    return obj["model"].predict_proba(design_matrix(X, obj["featset"], obj["columns"]).values)


@torch.no_grad()
def score_deep(name, cand, bs=4096):
    """Seed-ensembled final-loop probabilities [M,2] for every candidate."""
    ckpts = sorted((RESULTS / "deep" / name).glob("seed*_refit.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints for deep run {name}")
    t = load_tensors()
    rows = torch.as_tensor(cand.row.values, device=DEV)
    Qc = torch.as_tensor(query_features(cand), device=DEV)
    qc = torch.as_tensor(cand[["end_x", "end_y"]].values.astype(np.float32), device=DEV)
    zs = []
    for ck in ckpts:
        state = torch.load(ck, map_location=DEV, weights_only=False)
        cfg, meta = state["cfg"], state["meta"]
        model = build_model(cfg["model"], meta["f_player"], meta["f_query"], meta["f_ctx"], d=cfg["d"],
                            heads=cfg["heads"], loops=cfg["loops"], inject=cfg["inject"], geo=cfg["geo"],
                            drop=cfg["drop"]).to(DEV)
        model.load_state_dict(state["state_dict"]); model.eval()
        kw = {"n_loops": train_depth(cfg)} if cfg["model"] in ("loop", "loop_halt") else {}
        out = []
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            b = {k: t[k][r] for k in ["P", "ppos", "pteam", "pmask", "C"]}
            b.update(Q=Qc[i:i + bs], qpos=qc[i:i + bs])
            z = (model(**b, **kw) + model(**mirror(b), **kw)) / 2
            out.append(predict_probs(z)[1].cpu().numpy())
        zs.append(logit(np.concatenate(out)))
    return sigmoid(np.mean(zs, axis=0))


def is_deep(name):
    return (RESULTS / "deep" / name).exists() and not (MODELS / "tabular" / f"{name}__success.joblib").exists()


def score(name, task, cand, table, cache):
    if is_deep(name):
        if name not in cache:
            cache[name] = score_deep(name, cand)
        p = cache[name][:, TASK_COL[task]]
    else:
        p = score_tabular(name, task, cand, table)
    return np.clip(calibrator(name, task)(p), 1e-6, 1 - 1e-6)


def draw_pitch(ax):
    ax.set_facecolor("#2e7d32")
    for xy in [((0, 0), 120, 80), ((0, 18), 18, 44), ((102, 18), 18, 44), ((0, 30), 6, 20), ((114, 30), 6, 20)]:
        ax.add_patch(plt.Rectangle(*xy, fill=False, ec="white", lw=1))
    ax.plot([60, 60], [0, 80], c="white", lw=1)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, ec="white", lw=1))
    ax.set(xlim=(-2, 122), ylim=(82, -2), aspect="equal", xticks=[], yticks=[])


def plot_pass(pass_id, cands, passes, frames, out: Path):
    pr = passes.set_index("pass_id").loc[pass_id]
    f = frames[frames.pass_id == pass_id]
    c = cands[cands.pass_id == pass_id]
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_pitch(ax)
    opp, tm = f[~f.teammate], f[f.teammate]
    ax.scatter(opp.x, opp.y, c="#d32f2f", s=60, ec="k", zorder=3, label="opponent")
    ax.scatter(tm.x, tm.y, c="#1976d2", s=60, ec="k", zorder=3, label="teammate")
    best = c.loc[c.EV.idxmax()]
    for _, r in c.iterrows():
        col = "gold" if r.name == best.name else ("white" if r.is_actual else "#90caf9")
        ax.annotate("", (r.end_x, r.end_y), (pr.ball_x, pr.ball_y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.2 if r.is_actual or r.name == best.name else 0.8))
        ax.text(r.end_x + 1, r.end_y - 1.5, f"{r.EV:.3f}", color=col, fontsize=7, zorder=4)
    ax.scatter([pr.ball_x], [pr.ball_y], c="white", s=25, zorder=4)
    ax.set_title(f"{pr.player} ({pr.team}) {pr.minute}'  EV actual {c[c.is_actual == 1].EV.iloc[0]:.3f}"
                 f"  best {best.EV:.3f}  (white = actual, gold = best)", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-model", default="B2_DT")
    ap.add_argument("--q-model", default="B1_DT")
    ap.add_argument("--split", choices=["test", "train", "all"], default="test")
    ap.add_argument("--min-passes", type=int, default=200)
    ap.add_argument("--case-player", help="plot this player's largest-gap passes")
    ap.add_argument("--n-case", type=int, default=3)
    ap.add_argument("--plot-pass", nargs="*", default=[])
    args = ap.parse_args()

    table = load_model_table()
    cand = pd.read_parquet(TENSORS / "candidates.parquet")
    if args.split != "all":
        cand = cand[table.split.values[cand.row.values] == args.split].reset_index(drop=True)
    cache = {}
    cand["P"] = score(args.p_model, "success", cand, table, cache)
    cand["Q"] = score(args.q_model, "shot10", cand, table, cache)
    cand["EV"] = cand.P * cand.Q

    g = cand.groupby("row")
    best_idx = g.EV.idxmax()
    ps = pd.DataFrame({"row": best_idx.index,
                       "EV_best": cand.EV.values[best_idx.values],
                       "best_slot": cand.slot.values[best_idx.values],
                       "n_candidates": g.size().values})
    act = cand[cand.is_actual == 1].set_index("row")
    ps["EV_actual"] = act.EV.loc[ps.row].values
    ps["P_actual"], ps["Q_actual"] = act.P.loc[ps.row].values, act.Q.loc[ps.row].values
    ps["delta"] = ps.EV_best - ps.EV_actual
    ps["actual_is_best"] = ps.best_slot == -1
    meta_cols = ["pass_id", "match_id", "competition", "split", "team", "player", "minute", "y_success", "y_shot10"]
    ps = pd.concat([table.iloc[ps.row.values][meta_cols].reset_index(drop=True), ps.reset_index(drop=True)], axis=1)
    ps = ps[ps.n_candidates > 1]  # need at least one alternative

    out = RESULTS / "optimizer" / f"{args.p_model}+{args.q_model}"
    out.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out / "candidates_scored.parquet", index=False)
    ps.to_parquet(out / "passes_scored.parquet", index=False)

    def agg(by):
        a = ps.groupby(by).agg(passes=("delta", "size"), mean_delta=("delta", "mean"),
                               median_delta=("delta", "median"), argmax_share=("actual_is_best", "mean"),
                               mean_EV_actual=("EV_actual", "mean"))
        return a[a.passes >= args.min_passes].sort_values("mean_delta")
    agg("team").round(5).to_csv(out / "by_team.csv")
    agg(["player", "team"]).round(5).to_csv(out / "by_player.csv")

    summary = {"p_model": args.p_model, "q_model": args.q_model, "split": args.split, "passes": len(ps),
               "candidates_per_pass": float(ps.n_candidates.mean()),
               "actual_is_argmax_share": float(ps.actual_is_best.mean()),
               "delta_mean": float(ps.delta.mean()), "delta_median": float(ps.delta.median()),
               "delta_p90": float(ps.delta.quantile(0.9)), "EV_actual_mean": float(ps.EV_actual.mean()),
               "EV_best_mean": float(ps.EV_best.mean()),
               # sanity: EV of the actual target should track realised shots
               "shot10_rate_by_EV_actual_quintile": ps.groupby(pd.qcut(ps.EV_actual, 5, labels=False))
               .y_shot10.mean().round(4).tolist()}
    write_json(summary, out / "summary.json")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ps.delta[ps.delta > 0], bins=80, color="#1976d2")
    ax.set(yscale="log", xlabel="decision gap  EV(best) - EV(actual)", ylabel="passes (log)",
           title=f"Decision gap ({args.p_model} x {args.q_model}); actual = argmax for "
                 f"{summary['actual_is_argmax_share']:.1%}")
    fig.savefig(out / "decision_gap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    cases = list(args.plot_pass)
    if args.case_player:
        mine = ps[ps.player.str.contains(args.case_player, case=False)]
        print(f"\n{args.case_player}: {len(mine)} passes, mean delta {mine.delta.mean():.4f}, "
              f"argmax share {mine.actual_is_best.mean():.1%}")
        cases += mine.nlargest(args.n_case, "EV_actual").pass_id.tolist()
        cases += mine.nlargest(args.n_case, "delta").pass_id.tolist()
    if cases:
        frames = pd.read_parquet(PROC / "frames_long.parquet")
        frames = frames[frames.pass_id.isin(cases)]
        for pid in dict.fromkeys(cases):
            plot_pass(pid, cand, table, frames, out / f"case_{pid[:8]}.png")
        print(f"case-study figures: {len(set(cases))} in {out}")

    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()})


if __name__ == "__main__":
    main()
