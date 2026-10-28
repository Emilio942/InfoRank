import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def event_analysis(path, window=6):
    d = np.load(path, allow_pickle=True)
    R, dt, erank = d["R"], d["dt"], d["erank"]
    L = R.shape[1]
    n = (L - 1) // 2
    sl = slice(n - window, n + window + 1)

    R_align = R[:, sl] - R[:, [n - window - 1]]
    dt_align = dt[:, sl] - dt[:, [n - window - 1]]
    er_align = erank[:, sl]

    out = {
        "file": path.split("/")[-1],
        "L": L,
        "R_jump_at_sep": float(R_align[:, window + 1 :].mean() - R_align[:, : window + 1].mean()),
        "dt_jump_at_sep": float(dt_align[:, window + 1 :].mean() - dt_align[:, : window + 1].mean()),
        "erank_write": float(erank[:, 1:n].mean()),
        "erank_repeat": float(erank[:, n + 1 :].mean()),
        "dt_write": float(dt[:, 1:n].mean()),
        "dt_repeat": float(dt[:, n + 1 :].mean()),
        "R_write": float(R[:, 1:n].mean()),
        "R_repeat": float(R[:, n + 1 :].mean()),
    }
    t = stats.ttest_1samp(R_align[:, window + 1 :], R_align[:, : window + 1].mean(), axis=0)
    out["R_sep_ttest_p"] = float(t.pvalue.mean())
    return out, R_align.mean(0), dt_align.mean(0), er_align.mean(0), window


if __name__ == "__main__":
    files = sorted(glob.glob("results/geom_copy*.npz"))
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for f in files:
        row, Rm, dtm, erm, w = event_analysis(f)
        rows.append(row)
        lbl = row["file"].replace("geom_", "").replace(".npz", "")
        x = np.arange(-w, w + 1)
        axes[0].plot(x, Rm, marker="o", ms=2.5, lw=1, label=lbl)
        axes[1].plot(x, dtm, marker="o", ms=2.5, lw=1, label=lbl)
        axes[2].plot(x, erm, marker="o", ms=2.5, lw=1, label=lbl)
    for ax, ttl in zip(axes, ["delta scalar curvature R at SEP", "delta mean-step at SEP", "erank across SEP"]):
        ax.axvline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("token offset from SEP (write | repeat)")
        ax.set_title(ttl)
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig("results/fig_sep_event.png")
    plt.close(fig)
    with open("results/rq1_events.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    for r in rows:
        print(r)
