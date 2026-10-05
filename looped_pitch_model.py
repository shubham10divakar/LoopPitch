"""
LoopPitch: a weight-tied looped relational transformer for pass valuation.

One forward pass returns, per loop t = 1..T, two logits for the query (target) token:
    [:, t, 0] -> pass success   P(success | state, target)
    [:, t, 1] -> shot in 10 s   Q(shot | state, target, completed)

Token layout per sample (length L = 2 + N_MAX):
    0      context token   (match context, no pitch position)
    1      query token     (the pass target location: actual or hypothetical)
    2..    player tokens   (visible players from the 360 freeze-frame, padded)

NOTE: written against the design doc spec; run the smoke test at the bottom
(python looped_pitch_model.py) before training.
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

    def forward(self, x, bias, key_mask):
        B, L, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # [B,H,L,dk]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) + bias      # [B,H,L,L]
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        a = self.drop(att.softmax(dim=-1)) @ v                            # [B,H,L,dk]
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))


class LoopPitch(nn.Module):
    """
    tied=True,  loops=T  -> looped (weight-tied) transformer: 1 block applied T times
    tied=False, loops=T  -> standard T-layer transformer (ablation baseline)
    loops=1              -> single-layer set transformer (ablation baseline)
    inject=True          -> input injection: h_{t+1} = Block(h_t + e) for t >= 1
    """

    def __init__(self, f_player: int, f_query: int, f_ctx: int, d: int = 64, heads: int = 4,
                 loops: int = 4, tied: bool = True, inject: bool = True, drop: float = 0.1):
        super().__init__()
        self.loops, self.tied, self.inject = loops, tied, inject
        self.emb_p, self.emb_q, self.emb_c = mlp(f_player, d), mlp(f_query, d), mlp(f_ctx, d)
        self.type_emb = nn.Embedding(3, d)                    # 0 ctx, 1 query, 2 player
        self.blocks = nn.ModuleList([Block(d, heads, drop=drop) for _ in range(1 if tied else loops)])
        self.geo = GeoBias(heads)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))  # shared across loops

    def forward(self, P, ppos, pteam, pmask, Q, qpos, C, n_loops: int | None = None):
        """
        P [B,N,f_player] player features   ppos [B,N,2] raw coords
        pteam [B,N] 1=teammate 0=opponent  pmask [B,N] bool (True = real player)
        Q [B,f_query] target features      qpos [B,2] raw target coords
        C [B,f_ctx] context features
        n_loops: override T at test time (tied models only) for depth extrapolation
        returns logits [B,T,2]
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
        has_pos = key_mask.clone()
        has_pos[:, 0] = False                                             # context token has no position
        pos = torch.cat([torch.zeros(B, 1, 2, device=dev), qpos[:, None], ppos], dim=1)
        team = torch.cat([torch.full((B, 1), -1.0, device=dev),          # context
                          torch.ones(B, 1, device=dev),                   # target belongs to own team
                          pteam.float()], dim=1)
        bias = self.geo(pos, team, has_pos)

        h, outs = e, []
        for t in range(T):
            blk = self.blocks[0] if self.tied else self.blocks[t]
            h = blk(h + e if (self.inject and t > 0) else h, bias, key_mask)
            outs.append(self.head(h[:, 1]))                               # read out the query token
        return torch.stack(outs, dim=1)


def loop_weights(T: int, device=None):
    """Deep-supervision weights: later loops count more (linearly), sum to 1."""
    w = torch.arange(1, T + 1, dtype=torch.float, device=device)
    return w / w.sum()


def loss_fn(logits, y_succ, y_shot):
    """Success loss on all passes; shot loss only on completed passes. Deep-supervised over loops."""
    T = logits.size(1)
    w = loop_weights(T, logits.device)
    lp, lq = logits[..., 0], logits[..., 1]                               # [B,T]
    Ls = F.binary_cross_entropy_with_logits(lp, y_succ[:, None].expand_as(lp), reduction="none").mean(0)
    m = y_succ > 0.5
    if m.any():
        Lq = F.binary_cross_entropy_with_logits(lq[m], y_shot[m][:, None].expand_as(lq[m]),
                                                reduction="none").mean(0)
    else:
        Lq = torch.zeros_like(Ls)
    return ((Ls + Lq) * w).sum()


if __name__ == "__main__":  # smoke test with random tensors
    torch.manual_seed(0)
    B, N, FP, FQ, FC = 8, N_MAX, 10, 9, 20
    P, Q, C = torch.randn(B, N, FP), torch.randn(B, FQ), torch.randn(B, FC)
    ppos = torch.rand(B, N, 2) * torch.tensor([120.0, 80.0])
    qpos = torch.rand(B, 2) * torch.tensor([120.0, 80.0])
    pteam = torch.randint(0, 2, (B, N))
    pmask = torch.arange(N)[None] < torch.randint(6, N + 1, (B, 1))
    y_s, y_q = torch.randint(0, 2, (B,)).float(), torch.randint(0, 2, (B,)).float()
    for tied, T in [(True, 4), (False, 4), (True, 1)]:
        m = LoopPitch(FP, FQ, FC, loops=T, tied=tied)
        out = m(P, ppos, pteam, pmask, Q, qpos, C)
        loss = loss_fn(out, y_s, y_q); loss.backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"tied={tied} T={T} out={tuple(out.shape)} loss={loss.item():.4f} params={n:,}")
    m = LoopPitch(FP, FQ, FC, loops=4, tied=True).eval()
    print("test-time 8 loops:", tuple(m(P, ppos, pteam, pmask, Q, qpos, C, n_loops=8).shape))
