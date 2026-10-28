import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inforank.model import SSM
from inforank.tasks import sample_passkey, sample_copy, passkey_accuracy
from inforank.geometry import fisher_metric


def get_batch(args, model, device, B=None, L=None):
    B = B or args.batch
    L = L or args.L
    if args.task == "passkey":
        seq, vals, _ = sample_passkey(B, L, args.vocab, values=args.values, device=device)
        return seq, vals
    else:
        seq, _ = sample_copy(B, L, args.vocab, device=device)
        return seq, None


def metric_penalty(z, Mw, eps, tau):
    N, d = z.shape
    p = torch.softmax(z @ Mw.T, dim=-1)
    q = Mw.unsqueeze(0) * p.unsqueeze(2)
    g = torch.bmm(q.transpose(1, 2), Mw.unsqueeze(0).expand(N, -1, -1))
    w = z @ Mw.T @ Mw
    g = g - torch.bmm(w.unsqueeze(2), w.unsqueeze(1)) + eps * torch.eye(d, device=z.device)
    ev = torch.linalg.eigvalsh(g).clamp(min=1e-12)
    return ((ev.log() - math.log(tau)) ** 2).mean()


def evaluate(model, args, device, B=128, L=None):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(2):
            seq, vals = get_batch(args, model, device, B=B, L=L)
            logits = model(seq)
            total_loss += F.cross_entropy(logits[:, :-1].reshape(-1, args.vocab), seq[:, 1:].reshape(-1)).item()
            if args.task == "passkey":
                total_correct += (logits[:, -1].argmax(-1) == vals).sum().item()
                total_n += B
    model.train()
    out = {"val_loss": total_loss / 2}
    if args.task == "passkey":
        out["val_acc"] = total_correct / total_n
    else:
        n_rep = (seq.shape[1] - 1) // 2
        pred = logits.argmax(-1)
        tgt_rep = seq[:, n_rep + 1 :]
        pred_rep = pred[:, n_rep : n_rep + tgt_rep.shape[1]]
        m = tgt_rep >= 16
        out["val_acc"] = ((pred_rep == tgt_rep) & m).sum().item() / m.sum().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="passkey", choices=["passkey", "copy"])
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--values", type=int, default=8)
    p.add_argument("--recall-weight", type=float, default=64.0)
    p.add_argument("--freeze-timescales", action="store_true")
    p.add_argument("--reg-eps", type=float, default=0.0)
    p.add_argument("--reg-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/run")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    model = SSM(args.vocab, d=args.d, n=args.n, depth=args.depth).to(device)
    if args.freeze_timescales:
        for blk in model.blocks:
            blk.log_A.requires_grad_(False)
            blk.dt_bias.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / 200, 1.0)
        * (1.0 if s < 0.8 * args.steps else max(0.1, 1.0 - 0.9 * (s - 0.8 * args.steps) / (0.2 * args.steps))),
    )

    Mw = model.head.weight.detach()
    with torch.no_grad():
        seq, _ = get_batch(args, model, device, B=8)
        _, z0, _ = model(seq, return_states=True)
        sel = torch.randperm(z0.shape[1])[: 16].to(device)
        zf = z0[:, sel].reshape(-1, args.d)
        p = torch.softmax(zf @ Mw.T, dim=-1)
        q = Mw.unsqueeze(0) * p.unsqueeze(2)
        g = torch.bmm(q.transpose(1, 2), Mw.unsqueeze(0).expand(zf.shape[0], -1, -1))
        tau = (torch.diagonal(g, dim1=1, dim2=2).sum(-1) / args.d).median().clamp(min=1e-6).item()
    if args.reg_eps <= 0:
        args.reg_eps = 0.1 * tau
    print(f"tau (target metric scale) = {tau:.4f} reg_eps = {args.reg_eps:.2e}")

    best_acc, best_loss = -1.0, float("inf")
    log = []
    t0 = time.time()
    if args.task == "passkey":
        phases = [
            (max(32, args.L // 8), int(args.steps * 0.1)),
            (max(64, args.L // 4), int(args.steps * 0.25)),
            (max(128, args.L // 2), int(args.steps * 0.45)),
            (args.L, args.steps),
        ]
    else:
        phases = [
            (max(33, args.L // 8), int(args.steps * 0.1)),
            (max(65, args.L // 4), int(args.steps * 0.25)),
            (max(129, args.L // 2), int(args.steps * 0.45)),
            (args.L, args.steps),
        ]
    step = 0
    for L_phase, L_end in phases:
        while step < L_end:
            seq, vals = get_batch(args, model, device, L=L_phase)
            logits, z, dts = model(seq, return_states=True)
            tgt = seq[:, 1:]
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, args.vocab), tgt.reshape(-1), reduction="none"
            ).view(seq.shape[0], -1)
            if args.task == "passkey" and args.recall_weight > 1.0:
                w = 1.0 + (args.recall_weight - 1.0) * (tgt <= args.values).float()
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            if args.gamma > 0:
                B, Lc, d = z.shape
                idx = torch.randint(8, Lc - 1, (B, args.reg_k), device=device)
                zs = torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, d)).reshape(-1, d)
                loss = loss + args.gamma * metric_penalty(zs, Mw, args.reg_eps, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            last_dt = float(torch.stack([dt.detach().mean() for dt in dts]).mean())

            if step % 250 == 0 or step == args.steps - 1:
                ev = evaluate(model, args, device, B=64, L=L_phase)
                rec = {
                    "step": step,
                    "phase_L": L_phase,
                    "loss": loss.item(),
                    "grad_norm": float(gn),
                    **ev,
                    "dt_mean": last_dt,
                    "elapsed": time.time() - t0,
                }
                log.append(rec)
                print(rec, flush=True)
                if (ev.get("val_acc", -1) > best_acc) or (args.task == "copy" and ev["val_loss"] < best_loss):
                    best_acc = max(best_acc, ev.get("val_acc", -1))
                    best_loss = min(best_loss, ev["val_loss"])
                    torch.save(
                        {"model": model.state_dict(), "args": vars(args), "step": step},
                        args.out + ".pt",
                    )
            step += 1

    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, args.out + "_final.pt")
    with open(args.out + "_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print("DONE", args.out, "best_acc", best_acc, "best_loss", best_loss, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
