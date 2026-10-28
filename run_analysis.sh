#!/bin/zsh
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

for s in 0 1 2; do
  $PY extract.py --ckpt checkpoints/copy257_base_s${s}_final.pt --n-seq 16 --seed $s --out results/geom_copy257_base_s${s}.npz
done
$PY extract.py --ckpt checkpoints/copy257_reg01_s0_final.pt --n-seq 16 --seed 0 --out results/geom_copy257_reg01_s0.npz
$PY extract.py --ckpt checkpoints/copy257_reg01_s1_final.pt --n-seq 16 --seed 1 --out results/geom_copy257_reg01_s1.npz
$PY extract.py --ckpt checkpoints/copy513_base_s0_final.pt --n-seq 8 --seed 0 --out results/geom_copy513_base_s0.npz

$PY exp_len.py checkpoints/copy257_base_s0_final.pt checkpoints/copy257_base_s1_final.pt checkpoints/copy257_base_s2_final.pt checkpoints/copy257_reg01_s0_final.pt checkpoints/copy257_reg01_s1_final.pt checkpoints/copy513_base_s0_final.pt

$PY exp_rq1.py "results/geom_*.npz"
$PY exp_rq2.py "results/len_copy_*.npz"
$PY exp_rq3.py
$PY report_figs.py all
echo "ANALYSIS PIPELINE DONE"
