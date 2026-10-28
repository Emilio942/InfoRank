import glob
import json
import sys

import numpy as np
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/len_*.npz"))
    rows = []
    for path in files:
        d = np.load(path, allow_pickle=True)
        tag = path.split("/")[-1].replace(".npz", "")
        L = int(d["L"]) if "L" in d else d["erank"].shape[1]
        erank = d["erank"]
        rows.append(
            {
                "tag": tag,
                "L": L,
                "erank_end": float(erank[:, -32:].mean()),
                "erank_end_std": float(erank[:, -32:].std()),
                "erank_mean": float(erank.mean()),
                "lmin_end": float(d["lmin"][:, -32:].mean()),
                "logdet_end": float(d["logdet"][:, -32:].mean()),
                "cond_median": float(np.median(d["cond"])),
                "dt_end": float(d["dt"][:, -32:].mean()) if "dt" in d else None,
                "acc": float(d["acc"]) if "acc" in d else None,
            }
        )
        print(rows[-1])
    with open("results/rq2_summary.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)

    import re as _re
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        ck = _re.sub(r"_L\d+$", "", r["tag"])
        groups[ck].append(r)
    fits = {}
    for ck, rs in groups.items():
        rs = [r for r in rs if r["acc"] is not None]
        if len(rs) < 3:
            continue
        rs.sort(key=lambda r: r["L"])
        Ls = np.array([r["L"] for r in rs], float)
        E = np.array([r["erank_end"] for r in rs], float)
        A = np.array([r["acc"] for r in rs], float)
        beta, intercept, r, p, se = stats.linregress(Ls, E)
        b2, i2, r3, p3, _ = stats.linregress(Ls, A)
        r4, p4 = stats.pearsonr(A, E)
        C = np.array([r["cond_median"] for r in rs], float)
        LD = np.array([r["logdet_end"] for r in rs], float)
        r5, p5 = stats.pearsonr(A, np.log(C))
        r6, p6 = stats.pearsonr(A, LD)
        fits[ck] = {
            "beta_erank": beta, "r2_erank": r**2, "p_erank": p,
            "beta_acc": b2, "r2_acc": r3**2,
            "corr_acc_erank": r4, "p_corr": p4,
            "corr_acc_logcond": r5, "p_logcond": p5,
            "corr_acc_logdet": r6, "p_logdet": p6,
        }
        print(f"{ck}: erank_end = {intercept:.3f} + ({beta:.6f})·L  R2={r**2:.3f} (p={p:.1e}) | "
              f"acc slope={b2:.2e} R2={r3**2:.3f} | corr(acc, erank_end)={r4:.3f} (p={p4:.1e}) | "
              f"corr(acc, log cond)={r5:.3f} (p={p5:.1e}) | corr(acc, logdet)={r6:.3f} (p={p6:.1e})")
    with open("results/rq2_fits.json", "w") as fh:
        json.dump(fits, fh, indent=1)


if __name__ == "__main__":
    main()
