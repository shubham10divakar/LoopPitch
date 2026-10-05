# Data Cleaning Pipeline — Design Doc

**Project:** Possession value and optimal pass selection (Springer football chapter)
**Script:** `clean_pipeline.py` (single file, Python ≥ 3.9)
**Status:** Executed end to end on all 166 matches. Every number in this doc comes from that run.

---

## 1. Purpose

Turn raw StatsBomb event and 360 JSON into two analysis-ready tables:

| Output | Grain | Used by |
|---|---|---|
| `passes.parquet` | One row per open-play pass, with 30 features and 3 labels | Baselines B0–B5 (logistic regression, XGBoost, etc.) |
| `frames_long.parquet` | One row per visible player per pass | Optimizer (candidate receivers) and Phase 2 graph models |

Two supporting artefacts come out alongside: `drop_log.csv` (rows removed at each filter, per competition) and `qa_report.json` (label rates, orientation QA, missing values).

---

## 2. Data Size (measured)

### 2.1 Raw download

| Competition | Matches | Events JSON | 360 JSON | Total |
|---|---|---|---|---|
| Euro 2020 | 51 | 156 MB | 361 MB | 517 MB |
| WC 2022 | 64 | 191 MB | 439 MB | 630 MB |
| Euro 2024 | 51 | 153 MB | 381 MB | 534 MB |
| **Total** | **166** | **500 MB** | **1.18 GB** | **≈ 1.7 GB** |

Each match is about 10 MB, and the 360 frames take roughly 70% of that.

### 2.2 Processed output

| File | Rows | Columns | Size on disk | Size in memory |
|---|---|---|---|---|
| `passes.parquet` | **141,176** | 41 | 18 MB | ~62 MB |
| `frames_long.parquet` | **2,194,814** | 7 | 45 MB | ~156 MB |

**Hardware implications:**
- Everything fits comfortably in RAM on a laptop (under 0.5 GB peak).
- No GPU is needed for the baselines. XGBoost on 141k rows trains in seconds to a few minutes on CPU.
- The full pipeline ran in about 5 minutes, mostly download time; cached reruns take about 1 minute.
- Keep about 3 GB of free disk for the raw cache plus processed files.

---

## 3. Pipeline Stages

```
matches.json ──► ingest ──► per-match build ──► concat ──► write
                 (cache)    filter → label →            passes.parquet
                            features → QA               frames_long.parquet
                                                         drop_log.csv / qa_report.json
```

### Stage 1 — Ingest

- **Source:** the raw JSON on the StatsBomb open-data GitHub (`matches/`, `events/`, `three-sixty/`). This is the same source `statsbombpy` reads from, without the extra dependency.
- **Competition IDs:** Euro 2020 = (55, 43), WC 2022 = (43, 106), Euro 2024 = (55, 282).
- Keep only matches with `match_status_360 == "available"`. All 166 qualify.
- Cache to `data/raw/{competition}/{events|three-sixty}/{match_id}.json`. If a file already exists it is reused, which makes reruns idempotent.
- Retry 3 times with backoff on network errors.

### Stage 2 — Filter (in order, each step logged)

| # | Rule | Rationale |
|---|---|---|
| 1 | Drop set pieces: `pass.type` ∈ {Corner, Free Kick, Throw-in, Goal Kick, Kick Off} | Different spatial structure. **Keep `Recovery` and `Interception`**: those are open-play pass types, not set pieces. |
| 2 | Drop the penalty shootout (period 5) | Not open play |
| 3 | Drop outcomes "Injury Clearance" and "Unknown" | The first is a deliberate kick out of play; the second is unlabelled |
| 4 | Drop passes with no 360 frame | No spatial state to model |
| 5 | Drop start or end locations outside 0–120 × 0–80 | Corrupt coordinates |
| 6 | Drop duplicate event UUIDs | Hygiene |
| 7 | Drop frames with fewer than 6 in-bounds visible players | Too little context |

### Stage 3 — Labels

| Label | Definition |
|---|---|
| `y_success` | 1 if `pass.outcome` is absent (StatsBomb's convention for a completed pass), else 0 |
| `y_shot10` | 1 if a shot by the **same team** occurs in the **same period and possession**, **after** the pass, within **10.0 s** |
| `y_xg10` | The sum of `statsbomb_xg` over the shots in that window |

**Timing uses the `timestamp` field** (HH:MM:SS.mmm, which resets each period), not `minute`. Minutes continue counting across periods, so they would misplace second-half events. Event `index` order breaks ties when two events share a timestamp.

### Stage 4 — Features

- **State features (13):** `ball_x`, `ball_y`, `ball_dist_goal`, `ball_angle_goal`, `opp_within_5`, `nearest_opp_dist`, `opp_in_cone`, `def_line_x`, `tm_ahead`, `n_visible_teammates`, `n_visible_opponents`, `visible_area`, `under_pressure`
- **Action features (10):** `end_x`, `end_y`, `pass_len`, `pass_angle`, `progress_x`, `end_dist_goal`, `end_angle_goal`, `opp_near_target_3`, `lane_opp_2`, `target_in_box`
- **Context features:** `score_diff`, `period`, `minute`, `play_pattern`, `pass_type`, `body_part`, `pass_height`

Implementation notes:
- `ball_angle_goal` and `end_angle_goal` are the opening angle between the posts at (120, 36) and (120, 44).
- `opp_in_cone` counts opponents inside the triangle formed by the ball and the two posts.
- `lane_opp_2` counts opponents within 2 units of the segment from ball to target.
- `def_line_x` is the x position of the deepest visible **outfield** opponent, a proxy for the offside line.
- `visible_area` is the area of the shapely polygon built from the 360 `visible_area` vertex list.
- `score_diff` comes from a running tally of `Shot → Goal` events plus `Own Goal For` events, taken **before** each pass.

> **Never use as features:** `pass_recipient`, `shot_assist`, `goal_assist`, `assisted_shot_id`, or anything from later events. All of these leak the outcome.

### Stage 5 — QA checks

| Check | Result (full run) | Action |
|---|---|---|
| Actor present in frame | 141,176 / 141,176 | ✓ |
| Orientation: actor ≠ event location (> 2 units) | **34 rows (0.02%)** | Drop them, or flag them, before training |
| Missing values | 2 rows with no visible opponent, giving NaN `nearest_opp_dist` and `def_line_x` | Impute 99 and add a missing flag, or drop |
| Near-zero passes (`pass_len` < 0.5) | 28 rows | Inspect; probably drop |
| `score_diff` range | −7 to +7 | Valid: Spain beat Costa Rica 7–0 at WC 2022 |

Freeze-frame coordinates **use the same attacking-direction convention as events**, confirmed on 99.98% of rows. No flipping is needed.

---

## 4. Drop Log (full run)

| Step | Euro 2020 | WC 2022 | Euro 2024 | Total |
|---|---|---|---|---|
| All passes | 54,819 | 68,515 | 53,888 | **177,222** |
| − Set pieces | 4,744 | 6,263 | 4,447 | 15,454 |
| − Injury / unknown outcome | 283 | 350 | 216 | 849 |
| − No 360 frame | 5,909 | 7,595 | 5,405 | 18,909 |
| − Fewer than 6 visible players | 241 | 470 | 123 | 834 |
| **Kept** | **43,642** | **53,837** | **43,697** | **141,176 (79.7%)** |

The steps for out-of-bounds coordinates, duplicate IDs and the shootout removed 0 rows. Kept passes include 128,697 open-play, 11,264 Recovery and 1,215 Interception passes.

---

## 5. Final Dataset Profile

| Split | Passes | Pass success rate | Completed passes | `y_shot10` positives | `y_shot10` rate |
|---|---|---|---|---|---|
| Train (Euro 2020 + 2024) | 87,339 | 84.9% | 74,191 | 3,890 | 5.2% |
| Test (WC 2022) | 53,837 | 83.4% | 44,884 | 2,111 | 4.7% |
| **All** | **141,176** | **84.4%** | **119,075** | **6,001** | **5.0%** |

- That is about 850 kept passes per match (range 547–1,330).
- **Pass success** is mildly imbalanced (84 / 16). Use it as is.
- **Shot within 10 s** has about 5% positives. Report PR-AUC alongside ROC-AUC, calibrate the probabilities, and don't resample.
- The test set has a slightly lower success rate (83.4% vs 84.9%). Mention this distribution shift in the chapter.

---

## 6. Output Schemas

### `passes.parquet` (41 columns)

```
identity:  pass_id, match_id, competition, split, period, minute, second,
           team, player, possession
context:   play_pattern, pass_type, body_part, pass_height, under_pressure, score_diff
state:     ball_x, ball_y, ball_dist_goal, ball_angle_goal, opp_within_5,
           nearest_opp_dist, opp_in_cone, def_line_x, tm_ahead,
           n_visible_teammates, n_visible_opponents, visible_area
action:    end_x, end_y, pass_len, pass_angle, progress_x, end_dist_goal,
           end_angle_goal, opp_near_target_3, lane_opp_2, target_in_box
labels:    y_success, y_shot10, y_xg10
```

### `frames_long.parquet` (7 columns)

```
pass_id, slot, x, y, teammate, actor, keeper
```

To get the optimizer's candidate receivers for a pass, take its rows with `teammate & ~actor & ~keeper`.

---

## 7. Pre-training Checklist (feeds into the modelling stage)

1. Drop the 34 rows that fail the orientation check, the 28 near-zero passes, and the 2 rows with no visible opponent, or impute and flag them.
2. One-hot encode `play_pattern`, `pass_type`, `body_part` and `pass_height`.
3. Build GroupKFold folds by `match_id` on the training split only.
4. Train the pass-success model on all rows, and the action-value model on rows with `y_success == 1` only.
5. Don't touch WC 2022 until the final evaluation.

---

## 8. How to Run

```bash
pip install pandas numpy shapely pyarrow
python clean_pipeline.py --out data --limit 2   # smoke test (~6 matches, ~1 min)
python clean_pipeline.py --out data             # full run (~1.7 GB download, ~5 min)
```

Data © StatsBomb Open Data. Credit StatsBomb, with their logo, in any publication.
