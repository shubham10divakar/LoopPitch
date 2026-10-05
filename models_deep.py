"""
Deep baselines for the set / graph track. Same call signature and output shape as LoopPitch:
    model(P, ppos, pteam, pmask, Q, qpos, C) -> logits [B, 1, 2]   (success, shot10)

M2 DeepSets  phi(player, relative-to-target geometry) -> masked sum + max pool -> rho
M3 GAT       2 dense GATv2 layers over a k-NN player graph (k = 6) plus player<->query and
             context<->query edges; read out the query token
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from looped_pitch_model import LoopPitch, mlp


def rel_to_query(ppos, qpos):
    d = (ppos - qpos[:, None]) / 50.0
    return torch.cat([d, d.norm(dim=-1, keepdim=True)], dim=-1)          # [B,N,3]


class DeepSets(nn.Module):
    def __init__(self, f_player, f_query, f_ctx, d: int = 64, drop: float = 0.1, **_):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(f_player + 3, d), nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                 nn.Linear(d, d))
        self.emb_q, self.emb_c = mlp(f_query, d), mlp(f_ctx, d)
        self.rho = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d),
                                 nn.GELU(), nn.Linear(d, 2))

    def forward(self, P, ppos, pteam, pmask, Q, qpos, C, **_):
        h = self.phi(torch.cat([P, rel_to_query(ppos, qpos)], dim=-1))  # [B,N,d]
        m = pmask[..., None].float()
        s = (h * m).sum(1) / 11.0
        mx = h.masked_fill(~pmask[..., None], -1e4).max(1).values
        return self.rho(torch.cat([s, mx, self.emb_q(Q), self.emb_c(C)], dim=-1))[:, None]


class GATv2Layer(nn.Module):
    """Dense multi-head GATv2 with residual + pre-LN; adjacency [B,L,L] (row i attends to columns j)."""

    def __init__(self, d, heads=4, drop=0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.ws, self.wt = nn.Linear(d, d), nn.Linear(d, d)
        self.a = nn.Parameter(torch.randn(heads, self.dh) * self.dh ** -0.5)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x, adj):
        B, L, D = x.shape
        z = self.ln(x)
        s, t = self.ws(z).view(B, L, self.h, self.dh), self.wt(z).view(B, L, self.h, self.dh)
        e = F.leaky_relu(s[:, :, None] + t[:, None], 0.2)                # [B,L,L,H,dh]
        e = (e * self.a).sum(-1).permute(0, 3, 1, 2)                      # [B,H,L,L]
        att = e.masked_fill(~adj[:, None], float("-inf")).softmax(-1)
        msg = self.drop(att) @ t.permute(0, 2, 1, 3)                      # [B,H,L,dh]
        return x + self.drop(self.out(F.elu(msg.transpose(1, 2).reshape(B, L, D))))


class GAT(nn.Module):
    def __init__(self, f_player, f_query, f_ctx, d: int = 64, heads: int = 4, layers: int = 2, k: int = 6,
                 drop: float = 0.1, **_):
        super().__init__()
        self.k = k
        self.emb_p, self.emb_q, self.emb_c = mlp(f_player + 3, d), mlp(f_query, d), mlp(f_ctx, d)
        self.layers = nn.ModuleList([GATv2Layer(d, heads, drop) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def adjacency(self, ppos, pmask):
        B, N = pmask.shape
        dev = ppos.device
        d = torch.cdist(ppos, ppos).masked_fill(~pmask[:, None], float("inf"))
        knn = d.topk(min(self.k + 1, N), dim=-1, largest=False).indices      # includes self
        pp = torch.zeros(B, N, N, dtype=torch.bool, device=dev).scatter_(2, knn, True)
        pp = (pp | pp.transpose(1, 2)) & pmask[:, None] & pmask[:, :, None]
        adj = torch.zeros(B, N + 2, N + 2, dtype=torch.bool, device=dev)
        adj[:, 2:, 2:] = pp
        adj[:, 1, 2:] = pmask                                                # query <- players
        adj[:, 2:, 1] = pmask                                                # players <- query
        adj[:, 0, 1] = adj[:, 1, 0] = True                                   # context <-> query
        idx = torch.arange(N + 2, device=dev)
        adj[:, idx, idx] = True                                              # self loops
        return adj

    def forward(self, P, ppos, pteam, pmask, Q, qpos, C, **_):
        x = torch.cat([self.emb_c(C)[:, None], self.emb_q(Q)[:, None],
                       self.emb_p(torch.cat([P, rel_to_query(ppos, qpos)], dim=-1))], dim=1)
        adj = self.adjacency(ppos, pmask)
        for layer in self.layers:
            x = layer(x, adj)
        return self.head(x[:, 1])[:, None]


def build_model(name: str, f_player: int, f_query: int, f_ctx: int, **kw) -> nn.Module:
    """name in {deepsets, gat, tf, loop, loop_halt}; kw are passed through (d, heads, loops, inject, geo, ...)."""
    if name == "deepsets":
        return DeepSets(f_player, f_query, f_ctx, **kw)
    if name == "gat":
        return GAT(f_player, f_query, f_ctx, **kw)
    if name in ("tf", "loop", "loop_halt"):
        kw = {k: v for k, v in kw.items() if k in ("d", "heads", "loops", "inject", "geo", "drop")}
        return LoopPitch(f_player, f_query, f_ctx, tied=name != "tf", halt=name == "loop_halt", **kw)
    raise ValueError(name)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 4, 22
    args = (torch.randn(B, N, 10), torch.rand(B, N, 2) * 80, torch.randint(0, 2, (B, N)),
            torch.arange(N)[None] < torch.randint(6, N + 1, (B, 1)), torch.randn(B, 9), torch.rand(B, 2) * 80,
            torch.randn(B, 20))
    for name in ["deepsets", "gat", "tf", "loop", "loop_halt"]:
        m = build_model(name, 10, 9, 20)
        print(name, tuple(m(*args).shape), f"{sum(p.numel() for p in m.parameters()):,} params")
