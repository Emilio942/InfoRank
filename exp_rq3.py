import glob
import json
import sys

import numpy as np
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def final_metrics(log_path):
    log = json.load(open(log_path))
    tail = [r for r in log if r["step"] >= log[-1]["step"] - 250]
    accs = [r.get("val_acc") for r in tail if r.get("val_acc") is not None]
    losses = [r["val_loss"] for r in tail]
    return (float(np.mean(accs)) if accs else None, float(np.mean(losses)))


def main():
    rows = []
    for log_path in sorted(glob.glob("checkpoints/*_log.json")):
        tag = log_path.split("/")[-1].replace("_log.json", "")
        acc, loss = final_metrics(log_path)
        cfg = "reg" if "reg" in tag else "base"
        task = "copy" if "copy" in tag else "pk"
        gamma = tag.split("reg")[1].split("_s")[0] if "reg" in tag else "0"
        rows.append({"tag": tag, "task": task, "cfg": cfg, "gamma": gamma, "acc": acc, "val_loss": loss})
        print(rows[-1])
    with open("results/rq3_runs.json", "w") as f:
        json.dump(rows, f, indent=1)

    print("\npaired comparisons (final-250-step mean val_acc):")
    for task in ["copy", "pk"]:
        base = sorted([r["acc"] for r in rows if r["task"] == task and r["cfg"] == "base"])
        for g in sorted(set(r["gamma"] for r in rows if r["task"] == task and r["cfg"] == "reg")):
            reg = sorted([r["acc"] for r in rows if r["task"] == task and r["cfg"] == "reg" and r["gamma"] == g])
            if len(base) >= 2 and len(reg) >= 2:
                if len(base) == len(reg):
                    t, p = stats.ttest_rel(base, reg)
                else:
                    t, p = stats.ttest_ind(base, reg, equal_var=False)
                print(
                    f"{task} gamma={g}: base={np.mean(base):.4f} (n={len(base)}) reg={np.mean(reg):.4f} (n={len(reg)}) "
                    f"diff={np.mean(reg)-np.mean(base):+.4f} t_p={p:.3f}"
                )


if __name__ == "__main__":
    main()
