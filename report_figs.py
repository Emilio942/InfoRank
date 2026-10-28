import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/emilio/Documents/ai/InfoRank")

plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def fig_trajectory():
    fig, axes = plt.subplots(3, 2, figsize=(9, 6), sharex="col")
    for col, (pat, title) in enumerate([("geom_copy257_base_s0*", "copy L=257"), ("geom_copy513_base_s0*", "copy L=513")]):
        files = sorted(glob.glob(f"results/{pat}.npz"))
        d = np.load(files[0], allow_pickle=True)
        R, dt, erank = d["R"][0], d["dt"][0], d["erank"][0]
        t = np.arange(len(R))
        ax = axes[0, col]
        ax.plot(t, R, lw=0.6, color="firebrick")
        ax.set_ylabel("scalar curvature R")
        ax.set_title(title)
        ax = axes[1, col]
        ax.plot(t, dt, lw=0.6, color="teal")
        ax.set_ylabel("mean Delta_t")
        ax = axes[2, col]
        ax.plot(t, erank, lw=0.6, color="darkslateblue")
        ax.set_ylabel("erank(g_eps)")
        ax.set_xlabel("token position t")
        n = (len(R) - 1) // 2
        for ax in axes[:, col]:
            ax.axvline(n, color="gray", ls="--", lw=0.7)
            ax.axvline(len(R) - 1, color="black", ls=":", lw=0.7)
    fig.suptitle("Per-token geometric diagnostics along hidden-state trajectories (dashed: SEP boundary, dotted: end)")
    fig.tight_layout()
    fig.savefig("results/fig_trajectory.png")
    plt.close(fig)


def fig_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for i, pat in enumerate(["geom_copy257_base_s0*", "geom_copy257_base_s1*", "geom_copy257_base_s2*"]):
        f = sorted(glob.glob(f"results/{pat}.npz"))[0]
        d = np.load(f, allow_pickle=True)
        R, dt = d["R"][:, 1:].ravel(), d["dt"][:, 1:].ravel()
        idx = np.random.RandomState(0).choice(len(R), size=min(6000, len(R)), replace=False)
        ax = axes[i]
        ax.scatter(dt[idx], R[idx], s=1, alpha=0.15, color="firebrick")
        from scipy import stats as st
        r, p = st.pearsonr(R, dt)
        ax.set_title(f"seed {i}: rho={r:.3f} (p={p:.1e})")
        ax.set_xlabel("mean Delta_t")
        ax.set_ylabel("scalar curvature R")
    fig.tight_layout()
    fig.savefig("results/fig_scatter.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3))
    all_lags = {}
    for f in sorted(glob.glob("results/geom_copy257_base*.npz")):
        d = np.load(f, allow_pickle=True)
        R, dt = d["R"], d["dt"]
        for lag in range(-3, 4):
            a = R[:, max(0, -lag):R.shape[1] - max(0, lag) - 1]
            b = dt[:, max(0, lag) + 1:R.shape[1] - max(0, -lag)]
            n = min(a.shape[1], b.shape[1])
            r = np.corrcoef(a[:, :n].ravel(), b[:, :n].ravel())[0, 1]
            all_lags.setdefault(lag, []).append(r)
    lags = sorted(all_lags)
    mean = [np.mean(all_lags[l]) for l in lags]
    std = [np.std(all_lags[l]) for l in lags]
    ax.errorbar(lags, mean, yerr=std, fmt="o-", color="teal", capsize=3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lag (tokens; + means Delta leads R)")
    ax.set_ylabel("rho(R_t, Delta_{t+lag})")
    ax.set_title("Lagged coupling curvature <-> selection")
    fig.tight_layout()
    fig.savefig("results/fig_lag.png")
    plt.close(fig)


def fig_len():
    rows = json.load(open("results/len_sweep.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, task in [(axes[0], "copy"), (axes[1], "passkey")]:
        rr = [r for r in rows if r["task"] == task]
        by_ck = {}
        for r in rr:
            by_ck.setdefault(r["ckpt"], []).append(r)
        for ck, lst in by_ck.items():
            lst.sort(key=lambda r: r["L"])
            Ls = [r["L"] for r in lst]
            accs = [r["acc"] for r in lst]
            es = [r["erank_end"] for r in lst]
            ax2 = ax
            ax2.plot(Ls, accs, "o-", label=f"{ck} acc", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("context length L")
        ax.set_ylabel("task accuracy")
        ax.set_title(task)
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig("results/fig_len_acc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for ck in sorted(set(r["ckpt"] for r in rows)):
        lst = sorted([r for r in rows if r["ckpt"] == ck], key=lambda r: r["L"])
        ax.plot([r["L"] for r in lst], [r["erank_end"] for r in lst], "o-", label=ck, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("context length L")
    ax.set_ylabel("erank(g_eps) at final position")
    ax.set_title("Spectral collapse vs context length")
    ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig("results/fig_len_erank.png")
    plt.close(fig)


def fig_erank_trajectory():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for f, lab in [(sorted(glob.glob("results/geom_copy257_base_s0*.npz"))[0], "copy L=257"),
                   (sorted(glob.glob("results/geom_copy513_base_s0*.npz"))[0], "copy L=513")]:
        d = np.load(f, allow_pickle=True)
        e = d["erank"].mean(0)
        s = d["erank"].std(0) / np.sqrt(d["erank"].shape[0])
        ax.plot(e, label=lab, lw=1)
        ax.fill_between(np.arange(len(e)), e - s, e + s, alpha=0.25)
    ax.set_xlabel("token position t")
    ax.set_ylabel("erank(g_eps)")
    ax.set_title("Effective rank of the Fisher metric along context")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/fig_erank_traj.png")
    plt.close(fig)


def fig_reg():
    rows = json.load(open("results/rq3_runs.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, task in [(axes[0], "copy"), (axes[1], "pk")]:
        data = {}
        for r in rows:
            if r["task"] == task:
                data.setdefault(f"gamma={r['gamma']}", []).append(r["acc"])
        labels = list(data)
        ax.violinplot([data[k] for k in labels], showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("final val accuracy")
        ax.set_title(task)
    fig.tight_layout()
    fig.savefig("results/fig_reg.png")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "traj"):
        fig_trajectory()
    if which in ("all", "scatter"):
        fig_scatter()
    if which in ("all", "len"):
        fig_len()
    if which in ("all", "erank"):
        fig_erank_trajectory()
    if which in ("all", "reg"):
        fig_reg()
    print("figures done")
