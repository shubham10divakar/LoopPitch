"""
Set / graph track: M2 DeepSets, M3 GAT, M4 untied transformer, M5 LoopPitch, M6 LoopPitch + halting.

Protocol (design doc 6.3), per seed:
  1. 5-fold GroupKFold CV on the training split: early stopping on the held-out fold's
     validation loss (final-loop log-loss, success + shot on completed passes), patience 5.
     The held-out fold's predictions are the out-of-fold (OOF) predictions.
  2. Refit on the whole training split for round(mean best epoch) epochs; predict the test split.
  Mirror augmentation (y -> 80 - y, p = 0.5) in training; test-time average of both views.
  Tied models are evaluated at max(T, --eval-loops) loops, so every loop's logits are saved
  (loop t of the long run == the model run at depth t): this gives H2/H3 for free.

Outputs (results/deep/{name}/):
  seed{s}.parquet        per-loop logits (logitP_t*, logitQ_t*[, halt_t*]) for every pass
  seed{s}_refit.pt       refit checkpoint (config + state_dict)
  log_seed{s}.json       training curves, best epochs
and unified, seed-ensembled prediction files results/preds/{name}__{success,shot10}.parquet.

Usage:
    python train_deep.py --config configs/m5_looppitch.yaml
    python train_deep.py --model loop --name M5_T4 --loops 4 --seeds 0 1 2 3 4
    python train_deep.py --config configs/m5_looppitch.yaml --fast      # 1 fold instead of 5-fold CV
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from common import (N_FOLDS, PLAYER_Y_FLIP, PLAYER_Y_NEG, QUERY_Y_FLIP, RESULTS, TENSORS, load_model_table,
                    logit, save_preds, sigmoid, write_json)
from looped_pitch_model import loss_fn, predict_probs
from models_deep import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["P", "ppos", "pteam", "pmask", "Q", "qpos", "C"]

DEFAULTS = dict(model="loop", name=None, d=64, heads=4, loops=4, inject=True, geo=True, deep_sup=True,
                mirror=True, drop=0.1, lr=1e-3, wd=0.01, bs=512, epochs=80, patience=10, clip=1.0,
                seeds=[0], eval_loops=8, halt_beta=0.01, halt_prior=0.3, fast=False)


# ----------------------------------------------------------------------------- data
def load_tensors():
    z = np.load(TENSORS / "tensors.npz")
    t = {k: torch.from_numpy(z[k]) for k in z.files}
    t["pteam"] = t["pteam"].long()
    return {k: v.to(DEV) for k, v in t.items()}


def batch(t, idx):
    return {k: t[k][idx] for k in KEYS}, t["y_success"][idx], t["y_shot10"][idx]


def mirror(b, flip_mask=None):
    """Reflect the pitch vertically (y -> 80 - y) for the rows in flip_mask (all rows if None)."""
    b = dict(b)
    B = b["P"].size(0)
    m = torch.ones(B, dtype=torch.bool, device=b["P"].device) if flip_mask is None else flip_mask
    P, Q, ppos, qpos = b["P"].clone(), b["Q"].clone(), b["ppos"].clone(), b["qpos"].clone()
    pm = b["pmask"][m].float()
    for i in PLAYER_Y_FLIP:
        P[m, :, i] = (1 - P[m, :, i]) * pm
    for i in PLAYER_Y_NEG:
        P[m, :, i] = -P[m, :, i]
    for i in QUERY_Y_FLIP:
        Q[m, i] = 1 - Q[m, i]
    ppos[m, :, 1] = (80 - ppos[m, :, 1]) * pm
    qpos[m, 1] = 80 - qpos[m, 1]
    b.update(P=P, Q=Q, ppos=ppos, qpos=qpos)
    return b


# ----------------------------------------------------------------------------- train / predict
def make_model(cfg, meta):
    return build_model(cfg["model"], meta["f_player"], meta["f_query"], meta["f_ctx"], d=cfg["d"],
                       heads=cfg["heads"], loops=cfg["loops"], inject=cfg["inject"], geo=cfg["geo"],
                       drop=cfg["drop"]).to(DEV)


def is_tied(cfg):
    return cfg["model"] in ("loop", "loop_halt")


def train_depth(cfg):
    """Number of loops / layers the model produces logits for during training."""
    return cfg["loops"] if cfg["model"] in ("tf", "loop", "loop_halt") else 1


def eval_depth(cfg):
    """Tied models are also run deeper than trained (depth extrapolation, H3)."""
    return max(cfg["loops"], cfg["eval_loops"]) if is_tied(cfg) else train_depth(cfg)


@torch.no_grad()
def predict(model, t, idx, n_loops=None, tta=True, bs=2048):
    """Per-loop logits [n,T,C], averaged over the original and mirrored views."""
    model.eval()
    out = []
    kw = {"n_loops": n_loops} if n_loops else {}
    for i in range(0, len(idx), bs):
        b, _, _ = batch(t, idx[i:i + bs])
        z = model(**b, **kw)
        if tta:
            z = (z + model(**mirror(b), **kw)) / 2
        out.append(z.float().cpu())
    return torch.cat(out).numpy()


def val_loss(logits, ys, yq):
    """Final-prediction log-loss: success on all rows + shot10 on completed rows."""
    _, final = predict_probs(torch.from_numpy(logits))
    final = final.clamp(1e-6, 1 - 1e-6)
    ys, yq = torch.from_numpy(ys), torch.from_numpy(yq)
    m = ys > 0.5
    return (F.binary_cross_entropy(final[:, 0], ys) + F.binary_cross_entropy(final[m, 1], yq[m])).item()


def train(cfg, meta, t, tr_idx, va_idx=None, n_epochs=None, seed=0, tag=""):
    """Train with AdamW + cosine (1 epoch warm-up). Early-stops on va_idx if given, else runs n_epochs."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(cfg, meta)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_ep = math.ceil(len(tr_idx) / cfg["bs"])
    total = steps_per_ep * cfg["epochs"]

    def lr_at(step):
        if step < steps_per_ep:
            return (step + 1) / steps_per_ep
        return 0.5 * (1 + math.cos(math.pi * (step - steps_per_ep) / max(1, total - steps_per_ep)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    tr_idx_t = torch.as_tensor(tr_idx, device=DEV)
    ys_va = t["y_success"][va_idx].cpu().numpy() if va_idx is not None else None
    yq_va = t["y_shot10"][va_idx].cpu().numpy() if va_idx is not None else None
    best, best_ep, best_state, bad, hist = np.inf, 0, None, 0, []
    for ep in range(n_epochs or cfg["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=DEV)]
        for i in range(0, len(perm), cfg["bs"]):
            b, ys, yq = batch(t, perm[i:i + cfg["bs"]])
            if cfg["mirror"]:
                b = mirror(b, torch.rand(len(ys), device=DEV) < 0.5)
            loss = loss_fn(model(**b), ys, yq, deep_sup=cfg["deep_sup"], halt_beta=cfg["halt_beta"],
                           halt_prior=cfg["halt_prior"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step(); sched.step()
            tot += loss.item() * len(ys)
        rec = {"epoch": ep + 1, "train_loss": tot / len(perm), "sec": round(time.time() - t0, 1)}
        if va_idx is not None:
            rec["val_loss"] = val_loss(predict(model, t, va_idx, tta=False), ys_va, yq_va)
            if rec["val_loss"] < best - 1e-5:
                best, best_ep, bad = rec["val_loss"], ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        print(f"  {tag} ep {ep + 1:2d} train {rec['train_loss']:.4f}"
              + (f" val {rec['val_loss']:.4f}" if "val_loss" in rec else "") + f" ({rec['sec']}s)", flush=True)
        if va_idx is not None and bad >= cfg["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, hist


def logits_frame(z, prefix_cols):
    cols = {}
    for t_ in range(z.shape[1]):
        cols[f"logitP_t{t_ + 1}"] = z[:, t_, 0]
        cols[f"logitQ_t{t_ + 1}"] = z[:, t_, 1]
        if z.shape[2] == 3:
            cols[f"halt_t{t_ + 1}"] = z[:, t_, 2]
    _, final = predict_probs(torch.from_numpy(z))
    cols["pP"], cols["pQ"] = final[:, 0].numpy(), final[:, 1].numpy()
    return pd.DataFrame({**prefix_cols, **cols})


def final_probs_at(df, T, halting):
    """Final probabilities for a run evaluated at depth T (halting: expectation over the first T loops)."""
    z = np.stack([np.stack([df[f"logitP_t{t}"], df[f"logitQ_t{t}"]] +
                           ([df[f"halt_t{t}"]] if halting else []), -1) for t in range(1, T + 1)], 1)
    return predict_probs(torch.from_numpy(z.astype(np.float32)))[1].numpy()


# ----------------------------------------------------------------------------- driver
def run_seed(cfg, meta, t, p, seed, out):
    tr_all = np.flatnonzero(p.split.values == "train")
    te = np.flatnonzero(p.split.values == "test")
    folds = p.fold.values
    depth = eval_depth(cfg) if is_tied(cfg) else None
    z_all = np.full((len(p), eval_depth(cfg), 3 if cfg["model"] == "loop_halt" else 2), np.nan, np.float32)
    best_eps, log = [], {"cfg": cfg, "seed": seed, "folds": []}
    for k in ([0] if cfg["fast"] else range(N_FOLDS)):
        tr, va = tr_all[folds[tr_all] != k], tr_all[folds[tr_all] == k]
        model, be, hist = train(cfg, meta, t, tr, va, seed=seed, tag=f"[{cfg['name']} s{seed} f{k}]")
        best_eps.append(be)
        z_all[va] = predict(model, t, va, n_loops=depth)
        log["folds"].append({"fold": k, "best_epoch": be, "history": hist})
    n_ep = max(1, int(round(np.mean(best_eps))))
    if cfg["fast"]:  # reuse the fold-0 model on test instead of refitting
        final = model
    else:
        final, _, hist = train(cfg, meta, t, tr_all, n_epochs=n_ep, seed=seed, tag=f"[{cfg['name']} s{seed} refit]")
        log["refit"] = {"epochs": n_ep, "history": hist}
    z_all[te] = predict(final, t, te, n_loops=depth)
    torch.save({"cfg": cfg, "meta": meta, "state_dict": final.state_dict()}, out / f"seed{seed}_refit.pt")
    write_json(log, out / f"log_seed{seed}.json")
    df = logits_frame(z_all, {"pass_id": p.pass_id.values, "split": p.split.values, "fold": folds})
    df.to_parquet(out / f"seed{seed}.parquet", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, action=argparse.BooleanOptionalAction, default=None)
        elif isinstance(v, list):
            ap.add_argument(f"--{k}", dest=k, nargs="+", type=int, default=None)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=type(v) if v is not None else str, default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text()))
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    cfg["name"] = cfg["name"] or f"{cfg['model']}_T{cfg['loops']}"
    print(json.dumps(cfg), flush=True)

    meta = json.loads((TENSORS / "meta.json").read_text())
    p = load_model_table()
    t = load_tensors()
    out = RESULTS / "deep" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    n_params = sum(x.numel() for x in make_model(cfg, meta).parameters())
    print(f"{cfg['name']}: {n_params:,} parameters on {DEV}", flush=True)
    write_json({**cfg, "n_params": n_params}, out / "config.json")

    frames = [run_seed(cfg, meta, t, p, s, out) for s in cfg["seeds"]]

    # seed ensemble (mean logit of the final prediction at the trained depth) -> unified pred files
    halting = cfg["model"] == "loop_halt"
    ens = sigmoid(np.mean([logit(final_probs_at(f, train_depth(cfg), halting)) for f in frames], axis=0))
    base = {"pass_id": p.pass_id.values, "split": p.split.values, "fold": p.fold.values}
    ok = ~np.isnan(ens[:, 0])
    save_preds(pd.DataFrame({**base, "y": p.y_success.values, "p": ens[:, 0]})[ok], cfg["name"], "success")
    comp = ok & (p.y_success.values == 1)
    save_preds(pd.DataFrame({**base, "y": p.y_shot10.values, "p": ens[:, 1]})[comp], cfg["name"], "shot10")
    print(f"saved predictions for {cfg['name']} ({len(frames)} seeds)")


if __name__ == "__main__":
    main()
