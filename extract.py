import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inforank.model import SSM
from inforank.tasks import sample_passkey, sample_copy
from inforank.geometry import scalar_curvature, spectrum_stats, fisher_metric

DEV = "cuda"


def load_model(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    a = ck["args"]
    model = SSM(a["vocab"], d=a["d"], n=a["n"], depth=a["depth"]).to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, a


@torch.no_grad()
def rollout(model, seq):
    logits, z, dts = model(seq, return_states=True)
    return logits, z, dts


def extract(model, args, task, L, n_seq, seed, with_curvature, eps_rel=1e-3):
    V = args["vocab"]
    M = model.head.weight
    d = args["d"]
    seqs, metas = [], []
    for s in range(n_seq):
        if task == "passkey":
            seq, vals, pos = sample_passkey(1, L, V, device=DEV, seed=seed * 10000 + s)
            metas.append({"val": int(vals[0]), "pos": int(pos[0])})
        else:
            seq, _ = sample_copy(1, L, V, device=DEV, seed=seed * 10000 + s)
            metas.append({})
        seqs.append(seq)
    seq = torch.cat(seqs, 0)
    logits, z, dts = rollout(model, seq)
    dt = dts[0].mean(-1)

    Z = z.reshape(-1, d)
    with torch.no_grad():
        Zs = Z[:4096]
        p = torch.softmax(Zs @ M.T, dim=-1)
        gg = torch.bmm((M.unsqueeze(0) * p.unsqueeze(2)).transpose(1, 2), M.unsqueeze(0).expand(Zs.shape[0], -1, -1))
        scale = (torch.diagonal(gg, dim1=1, dim2=2).sum(-1) / d).median().item()
    eps = eps_rel * max(scale, 1e-9)

    out = {"dt": dt.cpu().numpy(), "eps": eps, "metric_scale": scale}
    with torch.no_grad():
        st = spectrum_stats(Z, M, eps)
    for k, v in st.items():
        out[k] = v.cpu().numpy().reshape(seq.shape[0], L)
    if with_curvature:
        torch.cuda.empty_cache()
        R = scalar_curvature(Z, M, eps, chunk=8)
        out["R"] = R.cpu().numpy().reshape(seq.shape[0], L)
    out["d_z"] = (z[:, 1:] - z[:, :-1]).norm(dim=-1).cpu().numpy()
    out["metas"] = json.dumps(metas)
    out["seq"] = seq.cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--n-seq", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--no-curvature", action="store_true")
    p.add_argument("--eps-rel", type=float, default=1e-3)
    args = p.parse_args()

    model, ma = load_model(args.ckpt)
    task = args.task or ma["task"]
    L = args.L or ma["L"]
    res = extract(model, ma, task, L, args.n_seq, args.seed, not args.no_curvature, eps_rel=args.eps_rel)
    res["args"] = json.dumps(ma)
    np.savez_compressed(args.out, **res)
    print("saved", args.out, "L=", L, "n_seq=", args.n_seq, "eps=", res["eps"], "scale=", res["metric_scale"])


if __name__ == "__main__":
    main()
