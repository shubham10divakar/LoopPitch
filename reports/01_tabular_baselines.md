# Step 1 — Tabular baselines (run 2026-10-05)

Command: `python train_tabular.py --trials 20` (RTX 3060, ~20 min).
Train: Euro 2020 + Euro 2024 (87,301 passes), out-of-fold (OOF) via 5-fold GroupKFold by match.
Test: WC 2022 (53,811 passes; 44,873 completed for shot10). Uncalibrated probabilities.
Raw numbers: [`tabular_summary.csv`](tabular_summary.csv); tuned XGBoost params: [`xgb_params/`](xgb_params/).

## Pass success (base rate 84.4%)

| Model | Features | Test log-loss ↓ | Brier ↓ | ROC-AUC ↑ | PR-AUC ↑ | ECE ↓ | BSS vs B0 ↑ |
|---|---|---|---|---|---|---|---|
| B0 prior | — | 0.4506 | 0.1388 | 0.500 | 0.834 | 0.016 | 0.000 |
| B0b grid 12×8 | locations | 0.3482 | 0.1070 | 0.828 | 0.953 | 0.027 | 0.229 |
| B3 XGB | locations | 0.2931 | 0.0895 | 0.883 | 0.971 | 0.014 | 0.355 |
| B1 logistic | DT | 0.2749 | 0.0853 | 0.908 | 0.979 | 0.025 | 0.385 |
| B1 logistic | FULL | 0.2623 | 0.0813 | 0.918 | 0.982 | 0.021 | 0.414 |
| M1 MLP | DT | 0.2490 | 0.0774 | 0.923 | 0.983 | 0.003 | 0.442 |
| **B2 XGB** | **DT** | **0.2277** | **0.0701** | **0.937** | **0.986** | 0.010 | **0.495** |
| B2 XGB | FULL | 0.2193 | 0.0674 | 0.942 | 0.988 | 0.010 | 0.514 |

## Shot within 10 s, completed passes (base rate 5.0%)

| Model | Features | Test log-loss ↓ | Brier ↓ | ROC-AUC ↑ | PR-AUC ↑ | ECE ↓ | BSS vs B0 ↑ |
|---|---|---|---|---|---|---|---|
| B0 prior | — | 0.1899 | 0.0448 | 0.500 | 0.047 | 0.005 | 0.000 |
| B0b grid 12×8 | locations | 0.1427 | 0.0361 | 0.847 | 0.364 | 0.005 | 0.195 |
| B3 XGB | locations | 0.1284 | 0.0340 | 0.891 | 0.426 | 0.004 | 0.241 |
| M1 MLP | DT | 0.1281 | 0.0342 | 0.895 | 0.427 | 0.003 | 0.237 |
| B1 logistic | DT | 0.1269 | 0.0339 | 0.896 | 0.434 | 0.002 | 0.244 |
| **B2 XGB** | **DT** | **0.1255** | **0.0336** | **0.900** | **0.440** | 0.003 | **0.250** |
| B2 XGB | FULL | 0.1251 | 0.0335 | 0.901 | 0.443 | 0.003 | 0.252 |

## Reading

1. **Pass success: the freeze-frame features matter a lot.** Going from locations only (B3) to decision-time
   features (B2_DT) cuts log-loss from 0.293 to 0.228 (−22%). Tuning improved B2_FULL over the untuned quick baseline
   (0.2208 → 0.2193).
2. **DT vs FULL costs little.** Removing post-pass information (body part, pass height) costs 0.008 log-loss on
   success and 0.0004 on shot10, so the optimizer-safe DT models are nearly as good as the FULL ones.
3. **XGBoost beats the MLP on tabular data** (0.228 vs 0.249), as the design doc expected.
4. **Shot within 10 s: the hand-crafted freeze-frame features add very little.** Log-loss goes from 0.1284
   (locations) to 0.1255 (DT), about 2%. Logistic regression is within 0.0014 of tuned XGBoost. This is the H4 gap
   the relational models (M4/M5) need to close.
5. **Context:** published 360 pass-success models report log-loss of about 0.17–0.27 (different data, so not
   directly comparable). B2 sits inside that range.
6. **Test log-loss is higher than OOF for success** (0.228 vs 0.206). WC 2022 has a lower success rate (83.4% vs
   84.9%), the distribution shift noted in the data doc. Calibration on OOF (step 4) will be checked against this.

**Targets for the deep models to beat (DT-equivalent inputs):** success < 0.2277, shot10 < 0.1255.
