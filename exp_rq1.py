import glob
import json
import sys

import numpy as np
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pearson(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[m], y[m])
    return r, p


def analyze_file(path, verbose=True):
    d = np.load(path, allow_pickle=True)
    R, dt = d["R"], d["dt"]
    erank, dz, eps = d["erank"], d["d_z"], float(d["eps"])
    L = R.shape[1]
    res = {"file": path, "L": L, "eps": eps, "n_seq": R.shape[0]}

    r, p = pearson(R[:, 1:], dt[:, 1:])
    res["rho_R_dt"] = r
    res["p_R_dt"] = p
    rs, ps = stats.spearmanr(R[:, 1:].ravel(), dt[:, 1:].ravel())
    res["rho_R_dt_spearman"] = rs

    r_c, p_c = pearson(R[:, 1:], dz)
    res["rho_R_dz"] = r_c
    res["p_R_dz"] = p_c
    r_dc, _ = pearson(dt[:, 1:], dz)
    res["rho_dt_dz"] = r_dc

    lags = range(-3, 4)
    lag_r = {}
    for lag in lags:
        if lag >= 0:
            a, b = R[:, 1:L - 1 + lag - lag] if False else R[:, 1:], dt[:, 1:]
        a, b = R[:, max(0, -lag):L - max(0, lag) - 1], dt[:, max(0, lag) + 1:L - max(0, -lag)]
        n = min(a.shape[1], b.shape[1])
        lag_r[lag] = pearson(a[:, :n], b[:, :n])[0]
    res["lagged_rho"] = {int(k): float(v) for k, v in lag_r.items()}

    flat_dt = dt[:, 1:].ravel()
    thr = np.quantile(flat_dt, 0.9)
    hi = R[:, 1:].ravel() >= np.quantile(R[:, 1:].ravel(), 0.9)
    sel = flat_dt >= thr
    res["dt_hi_R_mean"] = float(R[:, 1:].ravel()[sel].mean())
    res["dt_lo_R_mean"] = float(R[:, 1:].ravel()[~sel].mean())
    u, pu = stats.mannwhitneyu(R[:, 1:].ravel()[sel], R[:, 1:].ravel()[~sel])
    res["mw_p_dt_high_R"] = float(pu)

    L = R.shape[1]
    n = (L - 1) // 2
    res["phase_R_write"] = float(R[:, 1:n].mean())
    res["phase_R_repeat"] = float(R[:, n + 1 :].mean())
    res["phase_dt_write"] = float(dt[:, 1:n].mean())
    res["phase_dt_repeat"] = float(dt[:, n + 1 :].mean())
    res["phase_erank_write"] = float(erank[:, 1:n].mean())
    res["phase_erank_repeat"] = float(erank[:, n + 1 :].mean())
    rw, pw_ = pearson(R[:, 1:n], dt[:, 1:n])
    rr, pr_ = pearson(R[:, n + 1 :], dt[:, n + 1 :])
    res["rho_R_dt_write"] = rw
    res["p_R_dt_write"] = pw_
    res["rho_R_dt_repeat"] = rr
    res["p_R_dt_repeat"] = pr_

    res["R_mean"] = float(R.mean())
    res["R_std"] = float(R.std())
    res["erank_mean"] = float(erank.mean())
    res["erank_end_mean"] = float(erank[:, -32:].mean())
    res["erank_start_mean"] = float(erank[:, :32].mean())
    res["erank_drop"] = res["erank_start_mean"] - res["erank_end_mean"]
    res["lmin_mean"] = float(d["lmin"].mean())
    res["cond_median"] = float(np.median(d["cond"]))
    res["logdet_end"] = float(d["logdet"][:, -32:].mean())
    res["logdet_start"] = float(d["logdet"][:, :32].mean())

    if verbose:
        print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    out = {}
    for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/geom_*.npz")):
        res = analyze_file(path)
        out[path] = res
    with open("results/rq1_summary.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
