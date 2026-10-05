"""
LoopPitch: a weight-tied looped relational transformer for pass valuation.

One forward pass returns, per loop t = 1..T, two logits for the query (target) token:
    [:, t, 0] -> pass success   P(success | state, target)
    [:, t, 1] -> shot in 10 s   Q(shot | state, target, completed)
With halting (M6) a third channel [:, t, 2] holds the halting logit lambda_t.

Token layout per sample (length L = 2 + N_MAX):
    0      context token   (match context, no pitch position)
    1      query token     (the pass target location: actual or hypothetical)
    2..    player tokens   (visible players from the 360 freeze-frame, padded)

Smoke test: python looped_pitch_model.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MAX = 22  # max visible players in a frame


class GeoBias(nn.Module):
    """Pairwise geometry -> per-head additive attention bias.

    Distance (in pitch units) is expanded with Gaussian RBFs and combined with a
    same-team flag; a linear layer maps that to one scalar per head.
    Pairs involving the context token (no position) get zero bias.
    """

    def __init__(self, heads: int, n_rbf: int = 16, max_d: float = 60.0):
        super().__init__()
        self.register_buffer("mu", torch.linspace(0.0, max_d, n_rbf))
        spacing = max_d / (n_rbf - 1)
        self.gamma = 1.0 / (2 * spacing**2)
        self.proj = nn.Linear(n_rbf + 1, heads)

    def forward(self, pos, team, has_pos):
        # pos [B,L,2] raw pitch coords; team [B,L] (1 own, 0 opp, -1 none); has_pos [B,L] bool
        d = torch.cdist(pos, pos)                                         # [B,L,L]
        rbf = torch.exp(-self.gamma * (d.unsqueeze(-1) - self.mu) ** 2)   # [B,L,L,R]
        same = (team.unsqueeze(2) == team.unsqueeze(1)).float().unsqueeze(-1)
        b = self.proj(torch.cat([rbf, same], dim=-1))                     # [B,L,L,H]
        valid = (has_pos.unsqueeze(2) & has_pos.unsqueeze(1)).unsqueeze(-1).float()
        return (b * valid).permute(0, 3, 1, 2).contiguous()               # [B,H,L,L]


class Block(nn.Module):
    """Pre-LN transformer block with additive attention bias and key padding mask."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, drop: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias, key_mask, need_attn: bool = False):
        B, L, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # [B,H,L,dk]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) + bias      # [B,H,L,L]
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf")).softmax(dim=-1)
        a = self.drop(att) @ v                                            # [B,H,L,dk]
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return (x, att) if need_attn else x


def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


class LoopPitch(nn.Module):
    """
    tied=True,  loops=T  -> looped (weight-tied) transformer: 1 block applied T times   (M5)
    tied=False, loops=T  -> standard T-layer transformer (ablation baseline)            (M4)
    loops=1              -> single-layer set transformer (ablation baseline)
    inject=True          -> input injection: h_{t+1} = Block(h_t + e) for t >= 1
    geo=False            -> no geometric attention bias (ablation)
    halt=True            -> PonderNet-style halting head, adds logits[..., 2]           (M6)
    """

    def __init__(self, f_player: int, f_query: int, f_ctx: int, d: int = 64, heads: int = 4,
                 loops: int = 4, tied: bool = True, inject: bool = True, geo: bool = True,
                 halt: bool = False, drop: float = 0.1):
        super().__init__()
        self.loops, self.tied, self.inject, self.use_geo, self.halt = loops, tied, inject, geo, halt
        self.heads = heads
        self.emb_p, self.emb_q, self.emb_c = mlp(f_player, d), mlp(f_query, d), mlp(f_ctx, d)
        self.type_emb = nn.Embedding(3, d)                    # 0 ctx, 1 query, 2 player
        self.blocks = nn.ModuleList([Block(d, heads, drop=drop) for _ in range(1 if tied else loops)])
        self.geo = GeoBias(heads) if geo else None
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3 if halt else 2))  # shared across loops

    def forward(self, P, ppos, pteam, pmask, Q, qpos, C, n_loops: int | None = None, return_aux: bool = False):
        """
        P [B,N,f_player] player features   ppos [B,N,2] raw coords
        pteam [B,N] 1=teammate 0=opponent  pmask [B,N] bool (True = real player)
        Q [B,f_query] target features      qpos [B,2] raw target coords
        C [B,f_ctx] context features
        n_loops: override T at test time (tied models only) for depth extrapolation
        return_aux: also return {"hidden": [T x [B,L,d]], "attn": [T x [B,H,L,L]]}
        returns logits [B,T,2] (or [B,T,3] with halting)
        """
        T = n_loops or self.loops
        if not self.tied and T != self.loops:
            raise ValueError("untied model has a fixed depth")
        B, dev = P.size(0), P.device
        e = torch.cat([
            (self.emb_c(C) + self.type_emb.weight[0])[:, None],
            (self.emb_q(Q) + self.type_emb.weight[1])[:, None],
            self.emb_p(P) + self.type_emb.weight[2],
        ], dim=1)                                                         # [B,L,d]
        ones = torch.ones(B, 2, dtype=torch.bool, device=dev)
        key_mask = torch.cat([ones, pmask], dim=1)                        # [B,L]
        L = key_mask.size(1)
        if self.use_geo:
            has_pos = key_mask.clone()
            has_pos[:, 0] = False                                         # context token has no position
            pos = torch.cat([torch.zeros(B, 1, 2, device=dev), qpos[:, None], ppos], dim=1)
            team = torch.cat([torch.full((B, 1), -1.0, device=dev),      # context
                              torch.ones(B, 1, device=dev),               # target belongs to own team
                              pteam.float()], dim=1)
            bias = self.geo(pos, team, has_pos)
        else:
            bias = torch.zeros(B, self.heads, L, L, device=dev)

        h, outs, hidden, attn = e, [], [], []
        for t in range(T):
            blk = self.blocks[0] if self.tied else self.blocks[t]
            x = h + e if (self.inject and t > 0) else h
            if return_aux:
                h, a = blk(x, bias, key_mask, need_attn=True)
                hidden.append(h); attn.append(a)
            else:
                h = blk(x, bias, key_mask)
            outs.append(self.head(h[:, 1]))                               # read out the query token
        logits = torch.stack(outs, dim=1)
        return (logits, {"hidden": hidden, "attn": attn}) if return_aux else logits


def loop_weights(T: int, device=None, deep_sup: bool = True):
    """Deep-supervision weights: later loops count more (linearly), sum to 1. deep_sup=False -> last loop only."""
    if not deep_sup:
        w = torch.zeros(T, device=device); w[-1] = 1.0
        return w
    w = torch.arange(1, T + 1, dtype=torch.float, device=device)
    return w / w.sum()


def halting_dist(halt_logits):
    """PonderNet: p_t = lambda_t * prod_{s<t}(1 - lambda_s), with lambda_T forced to 1. [B,T] -> [B,T]."""
    lam = torch.sigmoid(halt_logits)
    lam = torch.cat([lam[:, :-1], torch.ones_like(lam[:, -1:])], dim=1)
    survive = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], dim=1), dim=1)
    return lam * survive


def _per_loop_bce(logits, y_succ, y_shot):
    """[B,T] success BCE on all passes and [B,T] shot BCE (zeros where not completed) + completed mask."""
    lp, lq = logits[..., 0], logits[..., 1]
    Ls = F.binary_cross_entropy_with_logits(lp, y_succ[:, None].expand_as(lp), reduction="none")
    Lq = F.binary_cross_entropy_with_logits(lq, y_shot[:, None].expand_as(lq), reduction="none")
    m = (y_succ > 0.5).float()[:, None]
    return Ls, Lq * m, m


def loss_fn(logits, y_succ, y_shot, deep_sup: bool = True, halt_beta: float = 0.01, halt_prior: float = 0.3):
    """Success loss on all passes; shot loss only on completed passes.

    Without halting: deep-supervised over loops with weights loop_weights(T).
    With halting (logits[..., 2]): expected loss under the halting distribution + beta * KL(p || Geometric(prior)).
    """
    T = logits.size(1)
    Ls, Lq, m = _per_loop_bce(logits, y_succ, y_shot)
    n_comp = m.sum().clamp(min=1.0)
    if logits.size(-1) == 2:
        w = loop_weights(T, logits.device, deep_sup)
        return (Ls.mean(0) * w).sum() + (Lq.sum(0) / n_comp * w).sum()
    p = halting_dist(logits[..., 2])                                      # [B,T]
    task = (p * Ls).sum(1).mean() + (p * Lq).sum() / n_comp
    k = torch.arange(T, device=logits.device, dtype=torch.float)
    prior = halt_prior * (1 - halt_prior) ** k
    prior = prior / prior.sum()
    kl = (p * (torch.log(p + 1e-9) - torch.log(prior))).sum(1).mean()
    return task + halt_beta * kl


def predict_probs(logits):
    """[B,T,2|3] logits -> [B,T,2] per-loop probabilities, and [B,2] final prediction.

    Final = last loop, or for halting models the expectation under the halting distribution.
    """
    pr = torch.sigmoid(logits[..., :2])
    if logits.size(-1) == 3:
        p = halting_dist(logits[..., 2])
        return pr, (p[..., None] * pr).sum(1)
    return pr, pr[:, -1]


if __name__ == "__main__":  # smoke test with random tensors
    torch.manual_seed(0)
    B, N, FP, FQ, FC = 8, N_MAX, 10, 9, 20
    P, Q, C = torch.randn(B, N, FP), torch.randn(B, FQ), torch.randn(B, FC)
    ppos = torch.rand(B, N, 2) * torch.tensor([120.0, 80.0])
    qpos = torch.rand(B, 2) * torch.tensor([120.0, 80.0])
    pteam = torch.randint(0, 2, (B, N))
    pmask = torch.arange(N)[None] < torch.randint(6, N + 1, (B, 1))
    y_s, y_q = torch.randint(0, 2, (B,)).float(), torch.randint(0, 2, (B,)).float()
    for kw in [dict(tied=True, loops=4), dict(tied=False, loops=4), dict(tied=True, loops=1),
               dict(tied=True, loops=4, geo=False), dict(tied=True, loops=4, halt=True)]:
        m = LoopPitch(FP, FQ, FC, **kw)
        out = m(P, ppos, pteam, pmask, Q, qpos, C)
        loss = loss_fn(out, y_s, y_q); loss.backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"{kw} out={tuple(out.shape)} loss={loss.item():.4f} params={n:,}")
    m = LoopPitch(FP, FQ, FC, loops=4, tied=True).eval()
    out, aux = m(P, ppos, pteam, pmask, Q, qpos, C, n_loops=8, return_aux=True)
    print("test-time 8 loops:", tuple(out.shape), "attn", tuple(aux["attn"][0].shape))
