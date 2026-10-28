# Empirical Report: Information-Geometric Diagnostics of a Selective SSM

**Date:** 2026-10-27 · **Hardware:** 1× RTX 3060 (12 GB) · **Runtime:** ~5 h end-to-end (incl. ablations)
**Code:** `inforank/` (model, tasks, geometry), `train.py`, `extract.py`, `exp_*.py`, `tests/validate_geometry.py`
**Artifacts:** `results/*.json`, `results/fig_*.png`, `checkpoints/*`

---

## 1. What was built

**Model.** Minimal Mamba-style selective SSM language model (`inforank/model.py`): input-dependent
discretization Δ_t = softplus(·), diagonal negative A (log_A frozen at init U[−8,−2]), low-rank input-dependent
B_t, C_t (n=16), SiLU gating, residual stream, RMSNorm + linear head. d=64, depth=2, V=64 (~230k params).
Parallel chunked scan (chunk 16, fp32-safe log-space recurrence), verified against a sequential reference
(max abs err 8×10⁻⁷).

**Task.** Copy task: sequence = S, SEP, S with S ∈ {16..63}^n. Next-token CE; report repeat-half accuracy.
Trained with length curriculum (L/8 → L/4 → L/2 → L). Checkpoints: `copy257_base_s{0,1,2}` (L=257),
`copy257_reg01_s{0,1}` (+ metric regularization), `copy513_base_s0` (L=513).

**Geometry.** Fisher metric of the softmax readout g(z)=Mᵀ(diag p − ppᵀ)M at the pre-head states z_t,
ε-regularized (ε = 10⁻³·tr(g)/d). Exact closed-form derivatives G1=∂g, G2=∂²g via softmax chain rule
(batched einsums, ~40 s for 16k points on the 3060). Christoffels → Riemann → Ricci → scalar curvature R
per token; spectrum stats (erank = exp(H(λ/Σλ)), λ_min, logdet, condition number).

**Validation (`tests/validate_geometry.py`, PASS).**
- Exact analytic test: full-simplex softmax readout ⇒ manifold isometric to round S^{n}: computed scalar
  curvature **5.000000** vs analytic n(n−1)/4 = 5 (err 5×10⁻⁷).
- AD vs float64 finite differences: rel. err 1×10⁻⁴; closed-form G1/G2 vs `jacrev`: err ~3×10⁻⁶.
- 1-D case: R ≡ 0.

**Methodological note (important).** An initial training matrix was invalid: `torch.Generator()` initializes
with the **constant** seed 67280421310721, so every `seed=None` sample returned identical data — the model
memorized one fixed batch and training-time eval scored memorized sequences (65% "accuracy" untrained-looking).
Fixed by deriving seeds from the global RNG; all results below are post-hoc on freshly seeded data and
training-time val_acc now matches post-hoc accuracy (e.g. 23.1% vs 23.2%).

---

## 2. RQ1 — Curvature singularities ↔ selection resets (Δ_t)

**Verdict: refuted as stated.** Pearson ρ(R_t, Δ_t) ≈ 0 in 4/6 models (|ρ| < 0.05, n.s.); −0.21 in one base
seed; +0.04 in the L=513 model. The hypothesis ρ > 0.3 fails everywhere. Lag structure is flat (|ρ(lag)| ≤ 0.21).

**ε-robustness (ablation).** Re-computing the pipeline at ε_rel ∈ {10⁻², 10⁻³, 10⁻⁴} (2 decades) on two models:
ρ(R, Δ) = −0.012 / −0.115 / +0.021 (copy257_s0) and +0.044 / +0.105 / −0.196 (copy513_s0) — small,
sign-unstable across ε, never near +0.3. The null result is not a regularization artifact; any apparent
correlation is ε-conditional. (Absolute |R| scales strongly with ε (e.g. mean −197 → −7300), so only
scale-free quantities — drops, correlations at fixed ε — are interpretable.) See `results/eps_ablation.json`.

**What the geometry *does* detect — the write→read phase switch.** The copy task has a hard task boundary
(SEP). Aligned across sequences (`results/fig_sep_event.png`, `results/rq1_events.json`):

| model | erank write → repeat | R write → repeat | Δ_t write → repeat |
|---|---|---|---|
| copy257_base_s0 | 43.6 → 15.8 | +209 → −584 | 0.0420 → 0.0420 |
| copy257_base_s2 | 44.8 → 14.5 | +490 → −271 | 0.0155 → 0.0155 |
| copy513_base_s0 | 41.3 → 24.6 | −2484 → −710 | 0.0412 → 0.0412 |

The **effective rank of the Fisher metric collapses by ~2–3× (44 → 16)** exactly at the memory-regime switch,
while the selection parameter Δ_t is *flat* across the boundary. The geometric observable localizes the
memory phase transition; the standard mechanistic variable (Δ) does not.

**Bootstrap (10k resamples, `results/sep_bootstrap.json`):** per-sequence erank drop at SEP is
+27.8 [27.5, 28.0], +13.9 [12.7, 15.0], +30.3 [29.7, 30.9] (base seeds), +16.7 [15.5, 17.8] (copy513_s0) —
every CI excludes zero decisively. The drop is **ε-invariant**: 27.65 / 27.70 / 27.63 and
19.20 / 19.24 / 19.18 across two decades of ε.

**Secondary structure.** ρ(R_t, ‖z_t − z_{t−1}‖) is strongly negative in base models (−0.28 … −0.66):
curvature peaks where the trajectory moves *slowly* — the state lingers in stiff (high-curvature) regions of
the readout manifold.

## 3. RQ2 — Spectral collapse with context length

**Verdict: the proposed erank operationalization is inverted; the collapse is real but appears in anisotropy.**
Models trained at L=257, swept over L ∈ {65 … 1025}:

- Within capacity (L ≤ 257): metric anisotropic — cond(g) ≈ 4×10³–9×10³, logdet ≈ −410…−460, erank_end ≈ 15–23.
- Beyond capacity (L ≥ 513, OOD): accuracy → ~0.5–4% and the metric **isotropizes**: cond → 1.0000,
  logdet plunges to −573 (= 64·log 1.3×10⁻⁴), erank → d = 64 exactly.

Consequently erank_end *increases* with L (positive slope, R² = 0.51–0.91) and correlates *negatively* with
accuracy (−0.30…−0.98, significant in 3/6 models): spectral-entropy erank is anti-informative here because
isotropic collapse inflates it to d. The physically meaningful capacity observables are
**cond(g) and logdet g**: accuracy correlates positively with log cond (up to r = 0.95, p = 0.013) and with
logdet (r = 0.90, p = 0.036 in the seed where the sweep has dynamic range).
Mechanistically: beyond capacity the readout saturates toward the noise marginal, the softmax covariance
Σ(p) loses directionality, and the latent statistical manifold degenerates — a geometric marker of the
information-preserving → dissipative transition the proposal set out to find.

**Replication.** The second L=513 model (`copy513_base_s1`, 22.4% at its training length) reproduces the
full pattern: erank_end = 9.7 + 0.046·L (R² = 0.82, p = 0.034), accuracy peaking at the trained length
(15.9% @ L=513) and collapsing to 1.1% @ L=1025 with erank → 63.7 ≈ d = 64.

## 4. RQ3 — Metric log-eigenvalue regularization

Penalty γ·E‖log g_ε − log(τI)‖²_F with τ = median tr(g)/d at init, γ = 0.1. Final val acc (mean over last
250 training steps, fresh seeded eval data):

| config | seeds | accuracy |
|---|---|---|
| copy257 base | 3 | 23.3% / 16.9% / 26.0% → **22.1%** |
| copy257 + reg γ=0.1 | 4 | 23.5% / 4.5% / 20.2% / 24.9% → **18.3%** |

**Verdict: no capability benefit (Welch t-test on the L=257 models: diff = −3.8 pp, p = 0.52).** One
regularized seed collapsed during training (4.5%); the others match the baseline. The penalty did achieve
its *stated geometric goal*: across the length sweep its metric is far more stable (erank_end std across L:
0.8–1.3 for reg vs 5–15 for base; the collapsed reg run holds ±0.8 over a 16× length range).
**Metric stability ≠ capability**: pinning log-eigenvalues removes exactly the anisotropy that carries task
information (cf. RQ2). With n=3 vs n=4 the absence of a positive effect is established at the
"no detectable benefit, possibly harmful" level, not as a precise effect-size estimate.

## 5. Figures

- `fig_trajectory.png` — per-token R, Δ_t, erank across a copy sequence (both anchor models)
- `fig_sep_event.png` — event-aligned R/Δ/erank at the SEP boundary (write→read switch)
- `fig_scatter.png`, `fig_lag.png` — R vs Δ scatter per seed; lagged correlations
- `fig_len_erank.png`, `fig_len_acc.png` — erank/accuracy vs context length
- `fig_reg.png` — base vs regularized accuracy

## 6. Honest caveats

- Small models (~230k params) on a synthetic task; "capacity" here is exact, measurable, and exhausted —
  but transfer to multi-layer pretrained Mamba LMs is untested.
- Scalar curvature is computed on the pre-head statistical manifold (readout-induced), not on the raw SSM
  state space; the proposal's M = WC is realized as the trained LM head over the normalized residual stream.
- Absolute curvature values are ε-dependent (they scale by ~10× per decade of ε); all reported conclusions
  use scale-free quantities (erank drops, correlations, condition numbers) and were verified across
  ε_rel ∈ {10⁻², 10⁻³, 10⁻⁴}.
- RQ3 is powered to rule out a benefit, not to estimate an effect size (n=3 vs n=4); the single collapsed
  reg seed may be a training-instability interaction worth its own study.

## 7. Reproduction

```bash
./run_training.sh     # ~65 min on the 3060
./run_analysis.sh     # ~25 min
.venv/bin/python tests/validate_geometry.py
```
