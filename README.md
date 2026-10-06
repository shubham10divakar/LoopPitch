# LoopPitch — Possession Value and Optimal Pass Selection from StatsBomb 360

Code for the football chapter: predict **pass success** P(success | state, target) and **shot within 10 s**
Q(shot | state, target, completed) from StatsBomb 360 freeze-frames, then score every visible teammate
as a hypothetical target (EV = P × Q) to measure each pass's **decision gap**.

The proposed model is **LoopPitch**: one weight-tied relational transformer block applied T times, with
a geometric attention bias, input injection, deep supervision across loops and test-time depth scaling.
It is compared against a ladder that runs from naive and classical baselines up to set and graph
models (design doc §4).

- Design doc: [`model_design_doc.md`](model_design_doc.md)
- Data pipeline doc: [`data_cleaning_pipeline.md`](data_cleaning_pipeline.md)
- Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). Credit StatsBomb in any publication.

---

## Quick start

```bash
pip install -r requirements.txt            # CUDA build of torch recommended

bash run_all.sh                            # P1: data -> baselines -> M4, M5 -> eval -> optimizer
bash run_all.sh p2                         #     + ablations
bash run_all.sh p3                         #     + DeepSets, GAT, adaptive halting
TRIALS=10 bash run_all.sh                  # quicker XGBoost tuning
```

Or run the steps one at a time:

```bash
python clean_pipeline.py --out data        # ~1.7 GB download (cached), ~5 min -> data/processed/
python build_tensors.py                    # ~1 min -> data/tensors/
python train_tabular.py --trials 50        # B0, B0b, B1, B2, B3, M1
python train_deep.py --config configs/m5_looppitch.yaml
python train_deep.py --config configs/m4_transformer.yaml
python calibrate.py
python evaluate.py
python optimize.py --p-model B2_DT --q-model B1_DT
python optimize.py --p-model M5_T4 --q-model M5_T4 --split all --case-player "Lamine Yamal"
```

Smoke tests: `python looped_pitch_model.py`, `python models_deep.py`, `python clean_pipeline.py --out data --limit 2`,
`python train_deep.py --model loop --name smoke --epochs 2 --fast`.

---

## Repository layout

| File | Role |
|---|---|
| `clean_pipeline.py` | StatsBomb JSON → `passes.parquet` (141,176 passes) + `frames_long.parquet` (2.19M player rows) |
| `common.py` | Paths, feature sets (DT / FULL / LOC), vectorised action features, cleaning, folds, metrics |
| `build_tensors.py` | Padded tensors `P[n,22,10] ppos pteam pmask Q[n,9] qpos C[n,18]`, plus the optimizer's **candidates** |
| `tabular_models.py` | B0 prior, B0b 12×8 grid lookup, B1 logistic, B2/B3 XGBoost + Optuna tuner, M1 MLP |
| `train_tabular.py` | Runs the tabular track: OOF predictions on train + refit → test |
| `looped_pitch_model.py` | **LoopPitch** (M4 untied / M5 tied / M6 halting), GeoBias, Block, deep-supervised loss |
| `models_deep.py` | M2 DeepSets, M3 GATv2, `build_model()` factory |
| `train_deep.py` | Deep track: CV + early stopping, refit, seeds, mirror augmentation + TTA, per-loop logits |
| `calibrators.py`, `calibrate.py` | Isotonic (trees) / temperature (neural) fitted on OOF predictions only |
| `evaluate.py` | Test metrics, ECE, match-bootstrap CIs, paired tests, reliability plots, per-loop analysis |
| `optimize.py` | EV for every candidate, decision gap Δ, team/player tables, case-study pitch plots |
| `analyze_loops.py` | Hidden-state convergence across loops (H3) and loop-wise attention maps on the pitch |
| `configs/*.yaml` | One file per deep experiment (M2–M6, ablations) |
| `run_all.sh` | End-to-end driver in priority order |
| `quick_baselines.py/.csv` | The original untuned baselines (design doc §3) |

Generated and git-ignored: `data/` (raw cache, processed parquet, tensors), `results/`, `logs/`.

```
results/
├── preds/{model}__{task}.parquet       pass_id, split, fold, y, p   (OOF on train, refit on test)
├── calibrated/{model}__{task}.parquet  + p_cal
├── deep/{run}/seed*.parquet            per-loop logits; seed*_refit.pt checkpoints; training logs
├── models/                             tabular models, XGB params, calibrators
├── metrics.csv  paired.csv  loops.csv  tabular_summary.csv
├── figures/                            reliability_*.png, loops_*.png
└── optimizer/{P}+{Q}/                  candidates_scored, passes_scored, by_team, by_player, case_*.png
```

---

## Protocol

| Item | Choice |
|---|---|
| Split | Train on Euro 2020 + Euro 2024 (87,301 passes), test on WC 2022 (53,811). The test split is used only for final predictions. |
| Cleaning | Pipeline checklist §7: drop 34 orientation mismatches, 28 near-zero passes and 2 frames with no opponent, leaving 141,112 passes |
| Folds | 5-fold `GroupKFold` by `match_id` inside train; stored in the `fold` column |
| Tasks | `success` on all passes (84.4% positive); `shot10` on completed passes only (5.0% positive) |
| Features | **DT** (decision-time): state + action + `play_pattern`, `pass_type`. **FULL** adds `body_part`, `pass_height` and is for prediction only, never the optimizer. **LOC**: the four coordinates. |
| Tabular | OOF predictions from the 5 folds; final model refit on all of train. XGBoost is tuned with Optuna (grouped CV, early stopping), then refit at the mean best iteration. |
| Deep | AdamW (lr 1e-3, wd 0.01), cosine schedule with 1-epoch warm-up, batch 512, ≤ 80 epochs, patience 10, grad-clip 1.0. CV early stopping, then refit at the mean best epoch. Several seeds, ensembled in logit space. |
| Augmentation | Vertical mirror y → 80 − y with p = 0.5; average of both views at test time |
| Calibration | Fitted on OOF predictions only: isotonic for trees, temperature for neural / linear models |
| Metrics | Log-loss (primary), Brier and BSS vs B0, ROC-AUC, PR-AUC, ECE (15 bins); 95% CIs from 1,000 bootstrap resamples **of matches**; paired bootstrap for model differences |

### Token features (LoopPitch / M2–M6)

- **Player (10):** x/120, y/80, teammate, actor, keeper, (x−bx)/50, (y−by)/50, dist_ball/50, dist_goal/120, angle_goal. Players are sorted by distance to the ball and padded to 22.
- **Query (9):** end_x/120, end_y/80, pass_len/50, progress_x/50, end_dist_goal/120, end_angle_goal, opp_near_target_3, lane_opp_2, target_in_box.
- **Context (18):** one-hot `play_pattern` (9) and `pass_type` (3), under_pressure, score_diff/3, minute/90, visible_area/8000, n_visible_teammates/11, n_visible_opponents/11.

Every query feature can be computed for a **hypothetical** target from the freeze-frame alone.
`build_tensors.py` checks that recomputing them for the actual target reproduces `passes.parquet` exactly.

### Candidates (optimizer)

For each pass, the candidate targets are the actual end location plus every visible teammate other
than the actor and keeper. That gives 1,057,548 candidates, about 6.5 teammates per pass. Δ = EV(best) − EV(actual) ≥ 0.

---

## Implementation notes and deviations from the design doc

- `pass_type` (Open Play / Recovery / Interception) is part of the context token and the DT set. It is known at decision time.
- **Training schedule (deviation):** the design doc's lr 3e-4 / 40 epochs / patience 5 left M5 undertrained (validation loss still falling at epoch 40 on every fold). On fold 0 of the training CV, lr 1e-3 / 80 epochs cut the validation loss from 0.346 to 0.316. Patience 10 because the validation loss is noisy (about ±0.003). Applied to all deep models alike; chosen on training folds only.
- **Early stopping:** the validation loss is the final-loop log-loss for success plus shot10 (shot10 on completed passes only).
- **Depth extrapolation (H3):** tied models are always run to `max(T, eval_loops=8)` loops, and the logits of every loop are saved. Loop t of that run is exactly the model run at depth t, so H2 and H3 need no extra inference.
- **M6 halting** is PonderNet-style: λ_T is forced to 1, the loss is the expected loss under the halting distribution + β·KL to a truncated geometric prior, and the prediction is the expected probability.
- **M3 GAT** is a dense GATv2 (no PyG dependency). Edges: k-NN among players (k = 6, symmetrised), player ↔ query, context ↔ query, and self-loops.
- **Not yet implemented:** B4 (un-xPass XGBoost) and B5 (SoccerMap). They need the external un-xPass repository. The design doc allows citing their published numbers instead.
- `--fast` in `train_deep.py` trains on folds 1–4 with fold 0 for early stopping, and skips the refit. Calibration then uses fold-0 OOF rows only. It is meant for quick ablation sweeps.

---

## Results

See [`RESULTS.md`](RESULTS.md); it is regenerated from `results/metrics.csv` after each full run.

### M5 vs tabular baselines (2026-10-06, calibrated, 95% match-bootstrap CIs)

M5_T4 (`configs/m5_looppitch.yaml`, 65,546 parameters, 5 seeds), test = WC 2022. Full tables are in `RESULTS.md`.

| Task | Model | Log-loss [95% CI] | Brier | ROC-AUC | ECE |
|---|---|---|---|---|---|
| success | **M5_T4** | **0.2146** [0.2042, 0.2249] | **0.0666** | **0.946** | 0.0101 |
| success | B2_FULL | 0.2191 [0.2087, 0.2297] | 0.0674 | 0.942 | 0.0083 |
| shot10 | **M5_T4** | **0.1241** [0.1158, 0.1337] | **0.0334** | **0.904** | 0.0032 |
| shot10 | B2_FULL | 0.1251 [0.1167, 0.1350] | 0.0336 | 0.901 | 0.0027 |

Paired match bootstrap, M5_T4 − B2_FULL log-loss: success −0.0045 [−0.0074, −0.0018], shot10 −0.0011 [−0.0020, −0.0002].
Both intervals exclude 0, so M5 beats the strongest tabular baseline on both tasks. The shot10 margin is small.

- **Loops (H2/H3):** success log-loss improves from 0.2221 at loop 1 to 0.2149 at loop 4, then stays flat out to loop 8 (no degradation when extrapolating depth).
  The loop 1→4 gain grows with congestion: 0.0033 with no opponent within 5 m, 0.0091 with one, 0.0121 with two or more.
- **Convergence:** the relative change of the token states falls from 1.04 at loop 2 to 0.095 at loop 12. It decays steadily but is not yet at a fixed point.
- **Calibration:** temperature T = 1.017 on both tasks, so M5 was already close to calibrated.

M4 (the untied control for H1) has not been run yet. Once it has, rerun `calibrate.py`, `evaluate.py` and `make_results.py`.

## Status

- [x] Data pipeline, pulled and verified (counts match `qa_report.json`)
- [x] Tensors, candidates, shared feature code
- [x] Tabular track B0, B0b, B1, B2, B3, M1
- [x] Deep track M2–M6 + ablation configs
- [x] Calibration, evaluation with bootstrap CIs, per-loop analysis
- [x] Optimizer + case studies
- [x] Loop convergence (H3) and loop-wise attention maps: `analyze_loops.py`
- [ ] B4 un-xPass, B5 SoccerMap
