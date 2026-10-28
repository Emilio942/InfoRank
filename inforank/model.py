import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveBlock(nn.Module):
    def __init__(self, d, n, dt_softplus=True):
        super().__init__()
        self.d, self.n = d, n
        self.in_proj = nn.Linear(d, d * 2 + n * 2 + d)
        self.out_proj = nn.Linear(d, d)
        with torch.no_grad():
            self.in_proj.weight.mul_(0.3)
        self.log_A = nn.Parameter(torch.empty(d, n).uniform_(-8.0, -2.0))
        self.dt_bias = nn.Parameter(torch.empty(d).uniform_(-6.0, -2.5))
        self.norm = nn.RMSNorm(d)

    def params(self, x):
        B, L, _ = x.shape
        z = self.in_proj(x)
        x_in, gate = z[..., :self.d], z[..., self.d:2 * self.d]
        dt = F.softplus(z[..., 2 * self.d:2 * self.d + self.d] + self.dt_bias).clamp(max=1.0)
        Bm, C = z[..., 2 * self.d + self.d:2 * self.d + self.d + self.n], z[..., 2 * self.d + self.d + self.n:]
        A = -torch.exp(self.log_A.clamp(max=-2.0))
        a = torch.exp(dt[..., :, None] * A[None, None])
        b = dt[..., :, None] * x_in[..., :, None] * Bm[..., None, :]
        return a, b, C, gate, dt

    @staticmethod
    def scan(a, b, chunk=16):
        B, L, d, n = a.shape
        pad = (-L) % chunk
        if pad:
            a = torch.cat([a, torch.ones(B, pad, d, n, device=a.device, dtype=a.dtype)], 1)
            b = torch.cat([b, torch.zeros(B, pad, d, n, device=a.device, dtype=a.dtype)], 1)
        Lp = a.shape[1]
        C = Lp // chunk
        a = a.view(B, C, chunk, d, n)
        b = b.view(B, C, chunk, d, n)
        P = torch.cumsum(torch.log(a.clamp(min=1e-30)), dim=2)
        cum = torch.cumsum(torch.exp(-P) * b, dim=2)
        h_chunk = torch.exp(P) * cum
        outs = []
        h_in = torch.zeros(B, d, n, device=a.device, dtype=a.dtype)
        for ci in range(C):
            h_out = torch.exp(P[:, ci]) * (h_in.unsqueeze(1) + cum[:, ci])
            outs.append(h_out)
            h_in = h_out[:, -1]
        hs = torch.stack(outs, dim=1).view(B, Lp, d, n)
        return hs[:, :L]

    def forward(self, x, return_dt=False):
        a, b, C, gate, dt = self.params(x)
        h = self.scan(a, b)
        y = (h * C[..., None, :]).sum(-1)
        y = y * F.silu(gate)
        out = x + self.out_proj(self.norm(y))
        return (out, dt) if return_dt else out


class SSM(nn.Module):
    def __init__(self, vocab, d=32, n=8, depth=2):
        super().__init__()
        self.vocab, self.d = vocab, d
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(SelectiveBlock(d, n) for _ in range(depth))
        self.final_norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.head.weight.mul_(0.5)

    def forward(self, tokens, return_states=False):
        x = self.embed(tokens)
        dts = []
        for blk in self.blocks:
            x, dt = blk(x, return_dt=True)
            dts.append(dt)
        z = self.final_norm(x)
        logits = self.head(z)
        if return_states:
            return logits, z, dts
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
