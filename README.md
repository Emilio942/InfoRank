# How I Burned Down My Research Proposal (3×) — and What You Can Learn From It

**InfoRank** · a *negative-result* study on Fisher-geometry diagnostics for selective state-space models

Research artifact accompanying the proposal *“Information-Geometric Phase Transitions in Selective
State-Space Models”*. We built the proposed pipeline end-to-end, validated it against exact analytic
results, trained small selective SSMs (Mamba-style) on memory tasks, and tested the three research
questions. **All three hypotheses were refuted** — and the refutations localize two robust,
replicable geometric phenomena that the proposal's observables missed.

This is a **negative-result study**: the value is in the *reasons* the hypotheses fail, which are
structural properties of the softmax–Fisher geometry (see [`WHY_IT_FAILED.tex`](WHY_IT_FAILED.tex)
for the mathematics and [`REPORT.md`](REPORT.md) for the full empirical report).

## Headline findings

1. **RQ1 refuted — no curvature↔Δ coupling.** Pearson ρ(scalar curvature, Δ_t) ≈ 0 (sign-unstable
   under ε-ablation over two decades). Root cause: curvature is a state function of the readout
   distribution p(h) alone — there is *no channel* from the dynamical gain Δ_t into the geometry;
   and the metric degenerates exponentially in the saturated limit (Σ(p) = O(δ) as p → one-hot),
   so confident events are geometrically invisible.
2. **RQ2 inverted — the capacity signal is anisotropy, not effective rank.** Beyond capacity the
   model outputs *confident garbage* (H(p) → 0), the state collapses onto an attractor, and
   g_ε → εI: condition number → 1, logdet plunges (~150 nats), and — because erank is
   scale-invariant — **erank rises to d** and is *anti*-correlated with accuracy (r up to −0.98).
   Use logdet/cond, not erank.
3. **RQ3 — no benefit (p = 0.52), and a tension.** The log-eigenvalue penalty stabilizes the metric
   (as designed) but rewards output uncertainty, opposes cross-entropy, and its minimizer (g = τI)
   destroys the diagnostic geometry itself.
4. **The survivor:** an ε-invariant, bootstrap-tight **effective-rank collapse of the Fisher metric
   at the write→read task boundary** (44 → 16, CIs excluding zero by orders of magnitude) while Δ_t
   stays flat — geometry sees the memory-regime switch that the selection parameter does not.

## Repository layout

```
inforank/            model (selective SSM + chunked parallel scan), tasks, geometry
train.py             training (curriculum, optional metric regularization)
extract.py           per-token trajectory extraction (states, Δ, curvature, spectra)
exp_rq1/2/3.py       hypothesis tests           exp_len.py     length sweeps
exp_events.py        SEP-boundary event study   exp_bootstrap.py  bootstrap CIs
report_figs.py       figures                    tests/validate_geometry.py  correctness tests
REPORT.md            full empirical report      WHY_IT_FAILED.tex  mathematical post-mortem
run_training.sh      reproduce all checkpoints  run_analysis.sh reproduce all results
```

## Requirements

Python ≥ 3.10, one CUDA GPU (≥ 6 GB free; developed on an RTX 3060 12 GB):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy matplotlib scipy
```

## Reproduce

```bash
./run_training.sh      # ~65 min on an RTX 3060
./run_analysis.sh      # ~25 min: extraction, stats, figures, summaries
.venv/bin/python tests/validate_geometry.py   # geometry correctness suite
```

## Correctness validation (in `tests/validate_geometry.py`)

- **Exact analytic test:** the full-simplex softmax Fisher manifold is isometric to a round sphere;
  the pipeline returns scalar curvature **5.000000** vs. analytic n(n−1)/4 = 5 (err 5×10⁻⁷).
- Closed-form metric derivatives vs. `jacrev`: err ~3×10⁻⁶; AD curvature vs. float64 finite
  differences: rel. err 10⁻⁴; 1-D case: R ≡ 0.

## Methodological warning for practitioners

`torch.Generator()` initializes with the **constant seed** `67280421310721`. Any sampling utility
that creates a fresh unseeded generator per call silently produces *identical data every call* —
models then memorize one fixed batch and training-time validation scores memorized sequences
(in our first, invalidated run: 65 % “accuracy” for an effectively untrained model). Seed such
generators explicitly from the global RNG; verify post-hoc accuracy on freshly seeded data.

## Scope & honesty

Tiny models (~230k params), synthetic copy task, 3–5 seeds per configuration. “Capacity” here is
exact and exhausted — transfer to pretrained Mamba LMs is untested. RQ3 is powered to rule out a
benefit, not to estimate an effect size. Full caveats in REPORT.md §6.

## License

None yet — add one before redistributing (e.g. MIT for code, CC-BY for the report).
