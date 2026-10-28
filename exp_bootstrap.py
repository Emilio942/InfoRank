import glob
import json

import numpy as np

rng = np.random.default_rng(0)


def bootstrap_ci(x, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


out = {}
for f in sorted(glob.glob("results/geom_copy*.npz")):
    d = np.load(f, allow_pickle=True)
    erank = d["erank"]
    L = erank.shape[1]
    n = (L - 1) // 2
    drop_per_seq = erank[:, 1:n].mean(1) - erank[:, n + 1 :].mean(1)
    m, lo, hi = bootstrap_ci(drop_per_seq)
    name = f.split("/")[-1].replace("geom_", "").replace(".npz", "")
    out[name] = {"erank_drop_mean": m, "ci95": [lo, hi], "n_seq": int(erank.shape[0])}
    print(f"{name:24s} erank drop = {m:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

with open("results/sep_bootstrap.json", "w") as fh:
    json.dump(out, fh, indent=1)
