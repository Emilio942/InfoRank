import json
import os
import sys

import numpy as np
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract import load_model
from inforank.tasks import sample_passkey, sample_copy
from inforank.geometry import spectrum_stats, fisher_metric

DEV = "cuda"


@torch.no_grad()
def copy_acc(model, V, L, B=128, seed=123):
    correct, total = 0, 0
    for s in range(4):
        seq, _ = sample_copy(B, L, V, device=DEV, seed=seed + s)
        pred = model(seq).argmax(-1)
        n = (L - 1) // 2
        tgt = seq[:, n + 1:]
        pr = pred[:, n:n + tgt.shape[1]]
        m = tgt >= 16
        correct += ((pr == tgt) & m).sum().item()
        total += m.sum().item()
    return correct / total


@torch.no_grad()
def passkey_acc(model, V, L, B=256, seed=123):
    correct = 0
    for s in range(2):
        seq, vals, pos = sample_passkey(B, L, V, device=DEV, seed=seed + s)
        pred = model(seq)[:, -1].argmax(-1)
        correct += (pred == vals).sum().item()
    return correct / (2 * B)


@torch.no_grad()
def spectrum_rollout(model, args, task, L, n_seq=8, seed=999):
    V, d = args["vocab"], args["d"]
    M = model.head.weight
    seqs = []
    for s in range(n_seq):
        if task == "passkey":
            seq, _, _ = sample_passkey(1, L, V, device=DEV, seed=seed * 100 + s)
        else:
            seq, _ = sample_copy(1, L, V, device=DEV, seed=seed * 100 + s)
        seqs.append(seq)
    seq = torch.cat(seqs, 0)
    _, z, dts = model(seq, return_states=True)
    Z = z.reshape(-1, d)
    Zs = Z[:2048]
    p = torch.softmax(Zs @ M.T, dim=-1)
    gg = torch.bmm((M.unsqueeze(0) * p.unsqueeze(2)).transpose(1, 2), M.unsqueeze(0).expand(Zs.shape[0], -1, -1))
    scale = (torch.diagonal(gg, dim1=1, dim2=2).sum(-1) / d).median().item()
    eps = 1e-3 * max(scale, 1e-9)
    st = spectrum_stats(Z, M, eps)
    out = {"L": L, "eps": eps}
    for k, v in st.items():
        out[k] = v.cpu().numpy().reshape(n_seq, L)
    out["dt"] = dts[0].mean(-1).cpu().numpy()
    return out


def main():
    ckpts = sys.argv[1:]
    summary = []
    for ck in ckpts:
        model, ma = load_model(ck)
        task = ma["task"]
        base = os.path.basename(ck).replace(".pt", "")
        acc_fn = passkey_acc if task == "passkey" else copy_acc
        if task == "passkey":
            Ls = [64, 128, 256, 384]
        else:
            Ls = [65, 129, 257, 513, 1025]
        for L in Ls:
            acc = acc_fn(model, ma["vocab"], L)
            spec = spectrum_rollout(model, ma, task, L)
            spec["acc"] = acc
            np.savez_compressed(f"results/len_{base}_L{L}.npz", **spec)
            row = {"ckpt": base, "task": task, "L": L, "acc": acc,
                   "erank_end": float(spec["erank"][:, -32:].mean()),
                   "erank_mean": float(spec["erank"].mean())}
            summary.append(row)
            print(row, flush=True)
    with open("results/len_sweep.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
