"""
Evaluation on the WC 2022 test split (design doc Section 7).

  metrics.csv        per model x task: log-loss, Brier, BSS vs B0, ROC-AUC, PR-AUC, ECE (raw and
                     calibrated), with 95% CIs from a bootstrap over MATCHES
  paired.csv         paired match-bootstrap of the log-loss difference vs a reference model
                     (default B2_DT), plus M5 vs B2_FULL and the H1 comparison M5 vs M4 when
                     they exist
  loops.csv          per-loop test log-loss for looped/untied runs (H2, H3), overall and stratified
                     by congestion (opp_within_5) and pitch third
  figures/           reliability diagrams per task, per-loop curves

Usage:
    python evaluate.py                 # 1,000 bootstrap resamples
    python evaluate.py --n-boot 200    # quicker
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from common import FIGS, RESULTS, load_model_table, metrics  # noqa: E402

CAL = RESULTS / "calibrated"


def load_test(path, table):
    df = pd.read_parquet(path)
    df = df[df.split == "test"].merge(table[["pass_id", "match_id"]], on="pass_id")
    return df


def match_boot_weights(match_ids, n_boot, seed=0):
    """[n_boot, n_rows] integer weights: each resample draws matches with replacement."""
    rng = np.random.default_rng(seed)
    um, inv = np.unique(match_ids, return_inverse=True)
    counts = rng.multinomial(len(um), np.full(len(um), 1 / len(um)), size=n_boot)  # [n_boot, n_matches]
    return counts[:, inv]


def boot_metrics(y, p, W):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = (p - y) ** 2
    out = {"logloss": (W @ ll) / W.sum(1), "brier": (W @ br) / W.sum(1)}
    out["auc"] = np.array([roc_auc_score(y, p, sample_weight=w) for w in W])
    out["prauc"] = np.array([average_precision_score(y, p, sample_weight=w) for w in W])
    return out


def reliability(ax, y, p, label, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax.plot(xs, ys, marker="o", ms=3, label=label)


def evaluate_models(table, n_boot, reference):
    rows, paired = [], []
    files = sorted(CAL.glob("*__*.parquet"))
    by_task = {}
    for f in files:
        name, task = f.stem.split("__")
        by_task.setdefault(task, {})[name] = load_test(f, table)

    for task, models in by_task.items():
        any_df = next(iter(models.values()))
        # B0 prior for the Brier skill score: the training base rate
        prior = pd.read_parquet(CAL / f"B0__{task}.parquet").p.iloc[0] if "B0" in models else None
        ids = np.sort(any_df.pass_id.unique())
        W = match_boot_weights(any_df.set_index("pass_id").loc[ids, "match_id"].values, n_boot)
        aligned = {n: d.set_index("pass_id").loc[ids] for n, d in models.items() if len(d) == len(ids)}
        for skipped in set(models) - set(aligned):
            print(f"  [{task}] {skipped}: test rows differ from the reference set; skipped")
        y = next(iter(aligned.values())).y.values
        fig, ax = plt.subplots(figsize=(5, 5))
        for name, d in aligned.items():
            for kind, col in [("raw", "p"), ("cal", "p_cal")]:
                r = metrics(y, d[col].values, prior)
                rec = {"task": task, "model": name, "probs": kind, **r}
                if kind == "cal":
                    bm = boot_metrics(y, d[col].values, W)
                    for k, v in bm.items():
                        rec[f"{k}_lo"], rec[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
                rows.append(rec)
            reliability(ax, y, d.p_cal.values, name)
        lim = 1.0 if task == "success" else 0.5
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="predicted", ylabel="observed",
               title=f"Reliability (calibrated) - {task}")
        ax.legend(fontsize=7)
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / f"reliability_{task}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        # paired bootstrap of log-loss differences (calibrated)
        def ll(d):
            p = np.clip(d.p_cal.values, 1e-6, 1 - 1e-6)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        pairs = [(n, reference) for n in aligned if n != reference and reference in aligned]
        for b in ["B2_FULL", "M4_T4"]:  # strongest tabular baseline; H1 control
            if "M5_T4" in aligned and b in aligned and b != reference:
                pairs.append(("M5_T4", b))
        for a, b in pairs:
            diff = ll(aligned[a]) - ll(aligned[b])
            bd = (W @ diff) / W.sum(1)
            paired.append({"task": task, "model": a, "vs": b, "d_logloss": diff.mean(),
                           "lo": np.percentile(bd, 2.5), "hi": np.percentile(bd, 97.5),
                           "p_better": float((bd < 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def evaluate_loops(table):
    """Per-loop log-loss on test for every deep run with more than one loop (seed-ensembled logits)."""
    rows = []
    tt = table.set_index("pass_id")
    for run in sorted((RESULTS / "deep").glob("*")):
        seeds = sorted(run.glob("seed*.parquet"))
        cfg_path = run / "config.json"
        if not seeds or not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        frames = [pd.read_parquet(s) for s in seeds]
        df = frames[0][frames[0].split == "test"][["pass_id"]].reset_index(drop=True)
        T = len([c for c in frames[0].columns if c.startswith("logitP_t")])
        if T < 2:
            continue
        info = tt.loc[df.pass_id]
        ys, yq = info.y_success.values, info.y_shot10.values
        congestion = pd.cut(info.opp_within_5.values, [-1, 0, 1, 99], labels=["0", "1", "2+"]).astype(str)
        third = pd.cut(info.ball_x.values, [-1, 40, 80, 121], labels=["def", "mid", "att"]).astype(str)
        for t in range(1, T + 1):
            zP = np.mean([f.loc[f.split == "test", f"logitP_t{t}"].values for f in frames], 0)
            zQ = np.mean([f.loc[f.split == "test", f"logitQ_t{t}"].values for f in frames], 0)
            pP, pQ = 1 / (1 + np.exp(-zP)), 1 / (1 + np.exp(-zQ))
            for strat, groups in [("all", np.full(len(df), "all")), ("opp_within_5", congestion),
                                  ("third", third)]:
                for g in np.unique(groups):
                    m = groups == g
                    mq = m & (ys == 1)
                    rows.append({"run": run.name, "trained_T": cfg["loops"], "loop": t, "strat": strat, "group": g,
                                 "n": int(m.sum()), "extrapolated": t > cfg["loops"],
                                 "logloss_success": metrics(ys[m], pP[m])["logloss"],
                                 "logloss_shot10": metrics(yq[mq], pQ[mq])["logloss"] if mq.sum() else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        for task in ("success", "shot10"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for run, d in df[df.strat == "all"].groupby("run"):
                ax.plot(d.loop, d[f"logloss_{task}"], marker="o", label=run)
                tr = d.trained_T.iloc[0]
                ax.axvline(tr, color="grey", lw=0.5, ls=":")
            ax.set(xlabel="loop t", ylabel="test log-loss (uncalibrated)", title=f"Per-loop log-loss - {task}")
            ax.legend(fontsize=7)
            fig.savefig(FIGS / f"loops_{task}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reference", default="B2_DT")
    args = ap.parse_args()
    table = load_model_table()

    m, paired = evaluate_models(table, args.n_boot, args.reference)
    m.round(4).to_csv(RESULTS / "metrics.csv", index=False)
    paired.round(5).to_csv(RESULTS / "paired.csv", index=False)
    loops = evaluate_loops(table)
    loops.round(4).to_csv(RESULTS / "loops.csv", index=False)

    pd.set_option("display.width", 200)
    cal = m[m.probs == "cal"]
    for task, d in cal.groupby("task"):
        print(f"\n=== {task} (test, calibrated; 95% match-bootstrap CI) ===")
        d = d.sort_values("logloss")
        print(d[["model", "logloss", "logloss_lo", "logloss_hi", "brier", "bss", "auc", "prauc", "ece"]]
              .round(4).to_string(index=False))
    if len(paired):
        print("\n=== paired log-loss differences (negative = better) ===")
        print(paired.round(4).to_string(index=False))
    if len(loops):
        print("\n=== per-loop test log-loss (all passes) ===")
        print(loops[loops.strat == "all"].pivot_table(index="loop", columns="run", values="logloss_success")
              .round(4).to_string())


if __name__ == "__main__":
    main()
