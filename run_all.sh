#!/usr/bin/env bash
# End-to-end pipeline in the design doc's priority order.
#   bash run_all.sh            # P1 only (data, baselines, M4, M5, evaluation, optimizer)
#   bash run_all.sh p2         # + ablations
#   bash run_all.sh p3         # + DeepSets, GAT, halting
set -euo pipefail
LEVEL=${1:-p1}
TRIALS=${TRIALS:-50}

# 1. data (cached: re-runs skip the download)
[ -f data/processed/frames_long.parquet ] || python clean_pipeline.py --out data
[ -f data/tensors/tensors.npz ] || python build_tensors.py

# 2. P1: tabular baselines + M4 / M5
python train_tabular.py --trials "$TRIALS"
python train_deep.py --config configs/m4_transformer.yaml
python train_deep.py --config configs/m5_looppitch.yaml

# 3. P2: ablations
if [ "$LEVEL" = p2 ] || [ "$LEVEL" = p3 ]; then
  for c in configs/ablation_*.yaml; do python train_deep.py --config "$c"; done
fi

# 4. P3: other deep baselines and adaptive halting
if [ "$LEVEL" = p3 ]; then
  for c in configs/m2_deepsets.yaml configs/m3_gat.yaml configs/m6_halting.yaml; do python train_deep.py --config "$c"; done
fi

# 5. calibration, evaluation, optimizer
python calibrate.py
python evaluate.py
python optimize.py --p-model B2_DT --q-model B1_DT
python optimize.py --p-model M5_T4 --q-model M5_T4 --split all --case-player "Lamine Yamal"
