# Model Design Doc — From Naive Baselines to a Looped Relational Transformer

**Chapter:** Possession value and optimal pass selection from StatsBomb 360 data
**Inputs:** `passes.parquet` (141,176 passes) and `frames_long.parquet` (2.19M player rows), from `clean_pipeline.py`
**Companion files:** `data_cleaning_pipeline.md`, `looped_pitch_model.py`
**Deadline:** 15 Oct 2026

---

## 0. One-Page Summary

| Item | Decision |
|---|---|
| Tasks | (1) **Pass success** P(success \| state, target). (2) **Shot within 10 s** Q(shot \| state, target, completed). |
| Downstream use | EV = P × Q for every visible teammate; the decision gap is EV(best) − EV(actual) |
| Model ladder | Naive → classical ML → published baselines → deep set/graph models → **LoopPitch** (looped transformer) |
| Proposed model | **LoopPitch**, a weight-tied relational transformer applied T times. It adds geometric attention bias, input injection, deep supervision across loops, and test-time depth scaling. |
| Split | Train on Euro 2020 + Euro 2024; test on WC 2022 (untouched until the end); GroupKFold by match inside train |
| Primary metric | Log-loss. Also Brier, ROC-AUC, PR-AUC, ECE. 95% CIs from a match-level bootstrap. |
| Must-have models | B0, B1, B2, B3, M1 (MLP), M4 (Transformer, untied), **M5 (LoopPitch)** |
| Nice-to-have models | B4 (un-xPass XGBoost), B5 (SoccerMap), M2 (DeepSets), M3 (GAT), M6 (adaptive halting) |

---

## 1. End-to-End Pipeline

```
StatsBomb open data (1.7 GB JSON)
        │  clean_pipeline.py  ✅ done
        ▼
passes.parquet (tabular) ─────────────┐        frames_long.parquet (players)
        │                             │                 │
        ▼                             │                 ▼
 Tabular track                        │         Set / graph track
 B0–B4, M1 (MLP)                      │         build_tensors.py → padded tensors
        │                             │         M2 DeepSets, M3 GAT, M4 Transformer,
        │                             │         M5 LoopPitch, M6 LoopPitch + halting
        ▼                             ▼                 ▼
               calibration (isotonic or temperature) on out-of-fold predictions
                                      ▼
                  evaluation on WC 2022 (metrics + bootstrap CIs)
                                      ▼
            optimizer: score every visible teammate → EV, best option, decision gap
                                      ▼
                  chapter tables, figures, case studies
```

---

## 2. Problem Formulation

For pass *i*, the state is s = {visible players with their roles} ∪ {context}, and the target a is the pass end location.

- **P task:** predict `y_success`. Trained on all 141k passes. Base rate 84.4%.
- **Q task:** predict `y_shot10`. Trained only on completed passes (119k). Base rate 5.0%.
- **Optimizer:** for each candidate c among the visible teammates, compute EV(s, c) = P̂(s, c) · Q̂(s, c).

**Inputs available for hypothetical targets.** Every target-dependent input must be computable from the freeze-frame plus a candidate location alone. That rules out pass height, body part, the recipient, and anything else that happens after the pass. Models used by the optimizer use the **"decision-time" feature set (DT)**. A "full" feature set (FULL) that also includes body part and pass height may be reported for prediction only.

---

## 3. Preliminary Results (already run)

These are untuned baselines from a single run: trained on Euro 2020 + 2024, tested on WC 2022, with no calibration step.

| Task | Model | Log-loss ↓ | Brier ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---|---|---|---|
| Pass success | B0 prior | 0.4507 | 0.1388 | 0.500 | 0.834 |
| | B1 logistic | 0.2624 | 0.0813 | 0.917 | 0.981 |
| | B3 XGB, location only | 0.2947 | 0.0900 | 0.882 | 0.971 |
| | **B2 XGB, all features** | **0.2208** | **0.0678** | **0.941** | **0.987** |
| Shot within 10 s (completed passes) | B0 prior | 0.1899 | 0.0448 | 0.500 | 0.047 |
| | **B1 logistic** | **0.1266** | **0.0338** | **0.897** | **0.436** |
| | B3 XGB, location only | 0.1302 | 0.0343 | 0.887 | 0.424 |
| | B2 XGB, all features | 0.1277 | 0.0341 | 0.896 | 0.430 |

What these say:

1. **Pass success:** the freeze-frame features matter a lot. Moving from location only to all features cuts log-loss from 0.295 to 0.221.
2. **Shot within 10 s:** hand-crafted freeze-frame features add almost nothing beyond location (0.130 → 0.128), and logistic regression is as good as XGBoost. This is the **open research question** the deep relational models address: can learned player-interaction representations pull out signal that hand-crafted counts miss?
3. For context only (different data, so not directly comparable): published 360 pass-success models report log-loss of roughly 0.17–0.27.

---

## 4. The Model Ladder

### Tier 0 — Naive (the floor)

| ID | Model | Input |
|---|---|---|
| B0 | Prior: the training base rate | none |
| B0b | Grid lookup: empirical rate for each (start zone × end zone) cell on a 12×8 grid, with Laplace smoothing | locations |

### Tier 1 — Classical ML (tabular, hand-crafted features)

| ID | Model | Notes |
|---|---|---|
| B1 | Logistic regression, L2 penalty | Standardized features, one-hot categoricals |
| B2 | XGBoost or LightGBM | Optuna, 50 trials, grouped CV |
| B3 | XGBoost on locations only | `ball_x`, `ball_y`, `end_x`, `end_y` (an xT-like comparison) |

### Tier 2 — Published baselines (re-run on our splits)

| ID | Model | Source |
|---|---|---|
| B4 | un-xPass XGBoost feature model | Open-source code from the un-xPass repository (ML-KULeuven/un-xPass) |
| B5 | SoccerMap: a fully convolutional network on a 104×68 surface | Same repository; the CNN baseline for 360 data |

Re-running these on identical splits makes them fair, citable comparisons. If time runs short, cite their reported numbers in related work instead.

### Tier 3 — Deep learning

| ID | Model | Input | Why include it |
|---|---|---|---|
| M1 | MLP: 3 × 128, ReLU, dropout 0.2 | Tabular DT features | Tests whether "deep" alone beats XGBoost on tabular data (usually it doesn't) |
| M2 | DeepSets: φ(player) → sum/max pool → ρ | Player set + query | The simplest permutation-invariant model |
| M3 | GAT, 2 layers: k-NN graph (k = 6) plus edges from each player to the query | Player graph | The standard GNN baseline (graph models in the literature: TacticAI, TGN) |
| M4 | Transformer, untied, T layers | Player tokens + query + context | The direct control for the looped model |

### Tier 4 — Proposed: LoopPitch (looped relational transformer)

| ID | Model |
|---|---|
| **M5** | **LoopPitch**: one weight-tied block applied T times, with geometric bias, input injection and deep supervision |
| M6 | LoopPitch + adaptive halting (PonderNet-style): the number of loops is chosen per pass. Stretch goal. |

---

## 5. LoopPitch — The Loop Idea

### 5.1 Intuition

Reading a football situation is **iterative**:
- Step 1: who is near the ball?
- Step 2: who is marking the potential receivers?
- Step 3: what space opens up after the reception?

A looped transformer applies the **same reasoning step** several times. Each loop lets information travel one more "hop" between players. Harder situations, such as a crowded final third, plausibly need more hops than easy ones, such as a free centre-back pass.

### 5.2 Architecture

```
tokens:  [CTX]  [QUERY = target location]  [P1 … P22 visible players, padded]
              │
      embeddings e = MLP(features) + type embedding
              │
   ┌──────────▼───────────┐
   │ Block_θ (shared)     │◄── input injection: h ← h + e (from loop 2 on)
   │  MHA + geometric bias │
   │  FFN, pre-LayerNorm   │──► head(h_QUERY) → [logit_P, logit_Q] at loop t
   └──────────┬───────────┘
              │ repeat T times (same θ)
              ▼
     final loop output → P̂, Q̂ (after calibration)
```

| Component | Specification |
|---|---|
| Player token (10 features) | x/120, y/80, teammate, actor, keeper, (x−bx)/50, (y−by)/50, dist_ball/50, dist_goal/120, angle_goal |
| Query token (9 features) | end_x/120, end_y/80, pass_len/50, progress_x/50, end_dist_goal/120, end_angle_goal, opp_near_target_3, lane_opp_2, target_in_box |
| Context token | One-hot play_pattern, under_pressure, score_diff/3, minute/90, visible_area/8000, n_visible_teammates/11, n_visible_opponents/11 |
| Geometric bias | Pairwise distance → 16 Gaussian RBFs (0–60 units) plus a same-team flag → linear map to one scalar per head, added to the attention logits. Pairs involving the context token get zero bias. |
| Padding | A key mask hides padded player slots. Every sample has at least 2 valid tokens (context and query), so no attention row is fully masked. |
| Block | Pre-LayerNorm, 4 heads, d = 64, FFN 4×, GELU, dropout 0.1 |
| Input injection | h_{t+1} = Block(h_t + e) for t ≥ 1. This keeps the raw input visible at every loop and stabilizes deep loops (as in looped-transformer work). |
| Readout | A shared head on the QUERY token at **every** loop, giving [B, T, 2] logits |
| Size | Tied T=4: **≈66k parameters**. Untied 4-layer (M4): **≈216k parameters**. Exact counts depend on the context one-hot width; the smoke test prints them. |

Implementation is in `looped_pitch_model.py` (the `LoopPitch` class). `tied=False` gives M4, and `loops=1` gives a single-layer set transformer.

### 5.3 Training objective

- L = Σ_t w_t · [ BCE(P_t, y_success) + BCE(Q_t, y_shot10) restricted to completed passes ]
- The loop weights w_t are proportional to t and normalized to sum to 1. This deep supervision makes **every** loop produce a usable prediction and lets you plot accuracy against the number of loops.
- **Augmentation:** mirror the pitch vertically (y → 80 − y) with probability 0.5 during training, and average both views at test time. TacticAI used the same reflection trick.

### 5.4 Hypotheses (these become the chapter's research questions)

| ID | Hypothesis | Test |
|---|---|---|
| H1 | At equal compute (same T), LoopPitch matches or beats the untied transformer with about 3× fewer parameters | M5 vs M4 at T = 4, with bootstrap CIs |
| H2 | More loops help more in congested states | Per-loop log-loss stratified by `opp_within_5` and pitch third |
| H3 | Test-time depth extrapolation: a model trained at T = 4 stays stable, or improves, when run at T = 6 or 8 | Evaluate at T ∈ {1, 2, 4, 6, 8}; plot convergence of ‖h_t − h_{t−1}‖ |
| H4 | The relational models beat hand-crafted features on the shot-within-10 s task, where XGBoost gains almost nothing from the freeze-frame | M4/M5 vs B2 on the Q task |

A **negative result** on H4 is still publishable: it would show that short-horizon threat in 360 data is mostly explained by location.

### 5.5 Ablations

| Ablation | Values |
|---|---|
| Loops T | 1, 2, 4, 8 |
| Weight tying | tied vs untied, at equal T (compute-matched) and at equal parameters (untied with d ≈ 36) |
| Input injection | on / off |
| Geometric bias | on / off (attention without distance information) |
| Deep supervision | all loops vs final loop only |
| Augmentation | mirror on / off |

---

## 6. Implementation Plan

### 6.1 Repository layout

```
football-chapter/
├── clean_pipeline.py            ✅ done
├── build_tensors.py             frames_long + passes → padded .pt/.npz tensors
├── train_tabular.py             B0–B4, M1  (sklearn / xgboost)
├── train_deep.py                M2–M6      (PyTorch; --model {deepsets,gat,tf,loop})
├── looped_pitch_model.py        ✅ LoopPitch, GeoBias, Block, loss_fn
├── calibrate.py                 isotonic / temperature scaling on OOF predictions
├── evaluate.py                  metrics, bootstrap CIs, reliability plots
├── optimize.py                  candidate EV, decision gap
├── configs/*.yaml               one per experiment
└── results/                     predictions_*.parquet, metrics.csv, figures/
```

### 6.2 `build_tensors.py` spec

- Group `frames_long` by `pass_id` and order the players by distance to the ball, so that if a frame is truncated, the nearest players are the ones kept.
- Pad to N_MAX = 22. Arrays: `P [n,22,10]`, `ppos [n,22,2]`, `pteam [n,22]`, `pmask [n,22]`, `Q [n,9]`, `qpos [n,2]`, `C [n,f_ctx]`, `y_success`, `y_shot10`, `match_id`, `split`.
- Size is about 141k × 22 × 14 floats ≈ 175 MB in float32. That fits in RAM.
- **Candidates file** for the optimizer: for every test pass, one query per visible teammate (excluding the actor and keeper), plus the actual target. Queries are about 54k × 8 ≈ 430k rows, and the player tensors are reused by index.

### 6.3 Training configuration (deep models)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, cosine schedule, 1 epoch warm-up |
| Batch / epochs | 512 / at most 40, early stopping on validation log-loss (patience 5) |
| Validation | 5-fold GroupKFold by match inside train. Report the CV mean, then refit on all of train for the test evaluation. |
| Seeds | 5 per model |
| Calibration | Temperature scaling for deep models and isotonic regression for trees, both fitted on out-of-fold predictions |
| Hardware | One Colab T4 or any GPU: about 1–2 minutes per epoch for M5. CPU is possible but slow for 5 seeds × ablations. |

### 6.4 Experiment matrix (priority order)

| Priority | Runs | Estimated compute |
|---|---|---|
| P1 | B0–B3 tuned + M1 | < 1 h on CPU |
| P1 | M4 (T = 4), M5 (T = 4), 5 seeds each | ~2–3 GPU hours |
| P1 | Optimizer on the best P model and best Q model | minutes |
| P2 | M5 ablations: T sweep, injection, bias, tying | ~3–4 GPU hours |
| P2 | B4 un-xPass XGBoost | ~1 h of setup |
| P3 | B5 SoccerMap, M2 DeepSets, M3 GAT, M6 halting | only if time remains |

---

## 7. Evaluation (same for every model)

- **Metrics:** log-loss (primary), Brier and Brier skill score vs B0, ROC-AUC, PR-AUC (essential for the 5% shot task), and ECE with 15 bins plus reliability diagrams.
- **Uncertainty:** 1,000 bootstrap resamples **of matches**, with a paired bootstrap for differences between models.
- **Interpretability:**
  - SHAP for B2.
  - For LoopPitch: attention maps from the QUERY token at each loop, overlaid on the pitch. This shows which defenders the model "checks" at loop 1 vs loop 4, and makes a strong chapter figure.
- **Downstream:**
  - The distribution of the decision gap Δ, and the share of passes where the actual choice was the argmax.
  - Δ per team and per player (minimum 200 passes).
  - Case studies, including the Yamal → Olmo cutback in the Euro 2024 final.

---

## 8. Chapter Mapping (book Parts 2 and 5)

| Chapter section | Content |
|---|---|
| Method: data | `data_cleaning_pipeline.md` |
| Method: models | The ladder (Section 4) and LoopPitch (Section 5) |
| Method: optimization | The EV argmax and decision gap; optionally a risk-adjusted λ-Pareto frontier |
| Results | Tables for P and Q, H1–H4, ablations, the decision-gap analysis |
| XAI | SHAP plus loop-wise attention maps on the pitch |
| GenAI (future work) | LLM-generated tactical reports from Δ and attention, evaluated for faithfulness |

---

## 9. Timeline (6–15 Oct)

| Date | Work |
|---|---|
| 6 Oct | `build_tensors.py`, `train_tabular.py` (tuned B0–B3, M1), calibration and evaluation scripts |
| 7 Oct | `train_deep.py`; smoke-test LoopPitch; M4 and M5 at 1 seed |
| 8 Oct | M4 and M5 at 5 seeds; first results table |
| 9 Oct | Ablations: T sweep, tying, injection, bias |
| 10 Oct | Optimizer, decision gap, attention figures, case studies |
| 11 Oct | B4 (un-xPass) if feasible; freeze all results |
| 12–14 Oct | Writing: methods, results, related work, discussion; Springer template |
| 15 Oct | Submit through CMT |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Deep models don't beat XGBoost on tabular-like data | Keep H4 framed as a question. The loop analyses (H2, H3) and interpretability still give the chapter its contribution. |
| GPU time | d = 64 and about 66k parameters keep it cheap; drop P3 runs first |
| Instability with many loops | Input injection, pre-LayerNorm, gradient clipping at 1.0, deep supervision |
| Selection bias in the optimizer (counterfactual targets were never actually chosen) | Discuss openly; restrict the analysis to visible teammates; report Δ as a relative measure |
| Overlap with other papers under review | Write fresh text for the method and cite your prior work where it is relevant |
