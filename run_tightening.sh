#!/bin/zsh
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for seed in 2 3; do
  $PY train.py --task copy --batch 64 --d 64 --n 16 --freeze-timescales --L 257 --steps 8000 --gamma 0.1 --seed $seed --out checkpoints/copy257_reg01_s$seed
done
$PY train.py --task copy --batch 64 --d 64 --n 16 --freeze-timescales --L 513 --steps 8000 --seed 1 --out checkpoints/copy513_base_s1
echo "TIGHTENING TRAINING DONE"
