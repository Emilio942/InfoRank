import sys
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inforank.geometry import (
    fisher_metric,
    metric_fn_factory,
    curv_from_derivs,
    scalar_curvature,
    fisher_metric_fd,
)
from torch.func import jacrev, jacfwd

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

V = 6
H = torch.eye(V) - torch.ones(V, V) / V
M = H[:, 1:].to(dev)
d = M.shape[1]
z = (0.3 * torch.randn(d, device=dev))
eps = 1e-8
R = scalar_curvature(z.unsqueeze(0), M, eps)[0].item()
expected = (V - 1) * (V - 2) / 4
print(f"[sphere test] V={V} computed R={R:.6f} analytic n(n-1)/4={expected:.6f} abs_err={abs(R-expected):.2e}")

z2 = 0.5 * torch.randn(d, device=dev)
R2 = scalar_curvature(z2.unsqueeze(0), M, eps)[0].item()
print(f"[sphere test 2] z2 computed R={R2:.6f} err={abs(R2-expected):.2e}")

Vf, df = 10, 4
Mf = torch.randn(Vf, df, device=dev)
zf = torch.randn(df, device=dev)
epsf = 1e-3
fn = metric_fn_factory(Mf, epsf)
g = fn(zf)
G1 = jacrev(fn)(zf)
G2 = jacrev(jacrev(fn))(zf)
R_ad = curv_from_derivs(g, G1, G2).item()
g_fd, G1_fd, G2_fd = fisher_metric_fd(zf.double(), Mf.double(), epsf, h=1e-2)
R_fd = curv_from_derivs(g_fd, G1_fd, G2_fd).item()
print(f"[fd check] d={df} V={Vf} R_ad={R_ad:.8f} R_fd={R_fd:.8f} rel_err={abs(R_ad-R_fd)/max(abs(R_fd),1e-12):.2e}")
print(f"[fd deriv check] G1 err={(G1.double()-G1_fd).abs().max():.2e} G2 err={(G2.double()-G2_fd).abs().max():.2e}")
g2_rel = ((G2.double() - G2_fd).abs().max() / G2.double().abs().max()).item()

M1 = torch.randn(12, 1, device=dev)
z1 = torch.randn(7, 1, device=dev)
R1 = scalar_curvature(z1, M1, 1e-3)
print(f"[1d test] max |R| = {R1.abs().max().item():.2e} (should be ~0)")

ok = (
    abs(R - expected) < 1e-3
    and abs(R2 - expected) < 1e-3
    and abs(R_ad - R_fd) < 5e-3 * max(1, abs(R_fd))
    and g2_rel < 5e-3
    and R1.abs().max() < 1e-4
)
print("VALIDATION", "PASS" if ok else "FAIL")
