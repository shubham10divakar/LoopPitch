"""
Loop-level analysis of a trained LoopPitch run (design doc 5.4 H3 and Section 7 interpretability).

  1. Convergence: relative change ||h_t - h_{t-1}|| / ||h_{t-1}|| of the token states per loop,
     on a sample of test passes, run past the trained depth (results/figures/{run}_convergence.png).
  2. Attention maps: attention from the QUERY token to every player at each loop, averaged over
     heads, drawn on the pitch (results/figures/{run}_attn_{pass}.png). Shows which defenders the
     model "checks" at loop 1 vs loop T.

Usage:
    python analyze_loops.py --run M5_T4                        # convergence + 3 high-EV test passes
    python analyze_loops.py --run M5_T4 --passes <pass_id> ...
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import FIGS, RESULTS, load_model_table, write_json  # noqa: E402
from models_deep import build_model  # noqa: E402
from optimize import draw_pitch  # noqa: E402
from train_deep import DEV, KEYS, load_tensors  # noqa: E402


def load_run(run, seed=0):
    state = torch.load(RESULTS / "deep" / run / f"seed{seed}_refit.pt", map_location=DEV, weights_only=False)
    cfg, meta = state["cfg"], state["meta"]
    m = build_model(cfg["model"], meta["f_player"], meta["f_query"], meta["f_ctx"], d=cfg["d"], heads=cfg["heads"],
                    loops=cfg["loops"], inject=cfg["inject"], geo=cfg["geo"], drop=cfg["drop"]).to(DEV)
    m.load_state_dict(state["state_dict"]); m.eval()
    return m, cfg


@torch.no_grad()
def convergence(model, cfg, t, idx, depth):
    rel = []
    for i in range(0, len(idx), 2048):
        b = {k: t[k][idx[i:i + 2048]] for k in KEYS}
        _, aux = model(**b, n_loops=depth, return_aux=True)
        valid = torch.cat([torch.ones_like(b["pmask"][:, :2]), b["pmask"]], 1).float()   # [B,L]
        hs = aux["hidden"]
        r = [((hs[k] - hs[k - 1]).norm(dim=-1) / (hs[k - 1].norm(dim=-1) + 1e-9) * valid).sum(1) / valid.sum(1)
             for k in range(1, depth)]
        rel.append(torch.stack(r, 1).cpu())
    return torch.cat(rel).mean(0).numpy()                                                 # [depth-1]


@torch.no_grad()
def attention_figure(model, cfg, t, row, info, out):
    b = {k: t[k][row:row + 1] for k in KEYS}
    T = cfg["loops"]
    _, aux = model(**b, n_loops=T, return_aux=True)
    pm = b["pmask"][0].cpu().numpy()
    pos = b["ppos"][0].cpu().numpy()[pm]
    team = b["pteam"][0].cpu().numpy()[pm]
    fig, axes = plt.subplots(1, T, figsize=(4.2 * T, 3.4))
    for k, ax in enumerate(np.atleast_1d(axes)):
        a = aux["attn"][k][0].mean(0)[1].cpu().numpy()          # heads-mean, row = QUERY token
        ap = a[2:][pm]
        draw_pitch(ax)
        ax.scatter(pos[:, 0], pos[:, 1], s=10 + 1500 * ap, c=np.where(team == 1, "#1976d2", "#d32f2f"),
                   ec="k", alpha=0.85, zorder=3)
        ax.annotate("", (info.end_x, info.end_y), (info.ball_x, info.ball_y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        ax.set_title(f"loop {k + 1}: query->ctx {a[0]:.2f}, self {a[1]:.2f}", fontsize=8)
    fig.suptitle(f"{info.player} ({info.team}) {info.minute}' - attention from the target token "
                 f"(marker size); blue = teammate, red = opponent", fontsize=9)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="M5_T4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12, help="loops for the convergence curve")
    ap.add_argument("--n-sample", type=int, default=10000)
    ap.add_argument("--passes", nargs="*", default=[])
    args = ap.parse_args()

    model, cfg = load_run(args.run, args.seed)
    t = load_tensors()
    p = load_model_table()
    FIGS.mkdir(parents=True, exist_ok=True)

    if cfg["model"] in ("loop", "loop_halt"):
        rng = np.random.default_rng(0)
        te = np.flatnonzero(p.split.values == "test")
        idx = torch.as_tensor(rng.choice(te, min(args.n_sample, len(te)), replace=False), device=DEV)
        rel = convergence(model, cfg, t, idx, args.depth)
        write_json({"run": args.run, "trained_T": cfg["loops"], "rel_change_by_loop": rel.round(5).tolist()},
                   RESULTS / f"convergence_{args.run}.json")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(range(2, args.depth + 1), rel, marker="o")
        ax.axvline(cfg["loops"], color="grey", ls=":", lw=1)
        ax.set(yscale="log", xlabel="loop t", ylabel="mean ||h_t - h_{t-1}|| / ||h_{t-1}||",
               title=f"{args.run}: state convergence (trained T = {cfg['loops']})")
        fig.savefig(FIGS / f"{args.run}_convergence.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        print("relative change per loop:", np.round(rel, 4))

    passes = list(args.passes)
    if not passes:  # default: three completed test passes into the box that led to a shot
        cand = p[(p.split == "test") & (p.y_shot10 == 1) & (p.target_in_box == 1) & (p.opp_within_5 >= 1)]
        passes = cand.sample(3, random_state=0).pass_id.tolist()
    rows = pd.Series(np.arange(len(p)), index=p.pass_id)
    for pid in passes:
        r = int(rows[pid])
        attention_figure(model, cfg, t, r, p.iloc[r], FIGS / f"{args.run}_attn_{pid[:8]}.png")
    print(f"attention figures for {len(passes)} passes in {FIGS}")


if __name__ == "__main__":
    main()
