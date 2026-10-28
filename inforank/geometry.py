import torch
from torch.func import vmap, jacrev


def fisher_metric(z, M, eps):
    p = torch.softmax(z @ M.T, dim=-1)
    Mp = M.T @ p
    g = (M * p.unsqueeze(-1)).T @ M - torch.outer(Mp, Mp)
    d = M.shape[1]
    return g + eps * torch.eye(d, device=z.device, dtype=z.dtype)


def metric_fn_factory(M, eps):
    def fn(z):
        return fisher_metric(z, M, eps)
    return fn


def curv_from_derivs(g, G1, G2):
    """g: (d,d), G1[a,b,c] = d_a g_bc, G2[a,b,c,d] = d_a d_b g_cd. Returns scalar R."""
    ginv = torch.linalg.inv(g)
    dginv = -torch.einsum("kp,apq,ql->akl", ginv, G1, ginv)
    base = G1.permute(1, 0, 2) + G1.permute(2, 0, 1) - G1
    Gamma = 0.5 * torch.einsum("kl,lij->kij", ginv, base)
    dGamma = 0.5 * (
        torch.einsum("akl,lij->akij", dginv, base)
        + torch.einsum("kl,ajli->akij", ginv, G2)
        + torch.einsum("kl,ailj->akij", ginv, G2)
        - torch.einsum("kl,alij->akij", ginv, G2)
    )
    t2 = dGamma.permute(3, 0, 1, 2)
    t3 = dGamma.permute(2, 0, 1, 3)
    Riem = (
        t2
        - t3
        + torch.einsum("lkm,mij->lijk", Gamma, Gamma)
        - torch.einsum("ljm,mik->lijk", Gamma, Gamma)
    )
    Ric = torch.diagonal(Riem, dim1=0, dim2=2).sum(-1)
    return torch.einsum("ij,ij->", ginv, Ric)


def metric_derivs_closed_form(Z, M, eps):
    """Z: (N,d) batched. Returns g (N,d,d), G1 (N,d,d,d), G2 (N,d,d,d,d) where
    G1[n,i,b,c] = d_i g_bc, G2[n,i,j,b,c] = d_i d_j g_bc, exact via softmax chain rule.
    p = softmax(Z M^T); g = M^T diag(p) M - (M^T p)(M^T p)^T + eps I."""
    N, d = Z.shape[0], M.shape[1]
    p = torch.softmax(Z @ M.T, dim=-1)
    w = p @ M
    q = torch.bmm(
        (M.unsqueeze(0) * p.unsqueeze(2)).transpose(1, 2), M.unsqueeze(0).expand(N, -1, -1)
    )
    eye = torch.eye(d, device=Z.device, dtype=Z.dtype)
    g = q - w.unsqueeze(2) * w.unsqueeze(1) + eps * eye
    D = M.unsqueeze(0) - w[:, None, :]
    H = q - w.unsqueeze(2) * w.unsqueeze(1)
    E = torch.einsum("nv,vb,nvi->nib", p, M, D)
    E2 = torch.einsum("nv,vb,nvi,nvj->nijb", p, M, D, D) - torch.einsum(
        "nij,nb->nijb", H, w
    )
    G1 = (
        torch.einsum("nv,vb,vc,nvi->nibc", p, M, M, D)
        - torch.einsum("nib,nc->nibc", E, w)
        - torch.einsum("nb,nic->nibc", w, E)
    )
    G2 = (
        torch.einsum("nv,vb,vc,nvi,nvj->nijbc", p, M, M, D, D)
        - torch.einsum("nij,nbc->nijbc", H, q)
        - torch.einsum("nijb,nc->nijbc", E2, w)
        - torch.einsum("nib,njc->nijbc", E, E)
        - torch.einsum("njb,nic->nijbc", E, E)
        - torch.einsum("nb,nijc->nijbc", w, E2)
    )
    return g, G1, G2


def scalar_curvature(Z, M, eps, chunk=64):
    outs = []
    with torch.no_grad():
        for i in range(0, Z.shape[0], chunk):
            g, G1, G2 = metric_derivs_closed_form(Z[i : i + chunk], M, eps)
            outs.append(vmap(curv_from_derivs)(g, G1, G2))
            if (i // chunk) % 100 == 99:
                torch.cuda.empty_cache()
    return torch.cat(outs)


def spectrum_stats(Z, M, eps, chunk=2048):
    fn = metric_fn_factory(M, eps)

    def stats(z):
        g = fn(z)
        ev = torch.linalg.eigvalsh(g)
        ev = torch.clamp(ev, min=1e-12)
        pn = ev / ev.sum()
        H = -(pn * torch.log(pn)).sum()
        return torch.stack(
            [torch.exp(H), ev[0], ev[-1], torch.log(ev).sum(), ev[-1] / ev[0].clamp(min=1e-30)]
        )

    outs = []
    for i in range(0, Z.shape[0], chunk):
        outs.append(vmap(stats)(Z[i : i + chunk]))
    S = torch.cat(outs)
    return {
        "erank": S[:, 0],
        "lmin": S[:, 1],
        "lmax": S[:, 2],
        "logdet": S[:, 3],
        "cond": S[:, 4],
    }


def fisher_metric_fd(z, M, eps, h=1e-4):
    d = z.shape[0]
    g = fisher_metric(z, M, eps)
    G1 = torch.zeros(d, d, d, dtype=g.dtype, device=g.device)
    for i in range(d):
        e = torch.zeros_like(z)
        e[i] = h
        G1[i] = (fisher_metric(z + e, M, eps) - fisher_metric(z - e, M, eps)) / (2 * h)
    G2 = torch.zeros(d, d, d, d, dtype=g.dtype, device=g.device)
    for i in range(d):
        for j in range(d):
            ei = torch.zeros_like(z)
            ej = torch.zeros_like(z)
            ei[i] = h
            ej[j] = h
            G2[i, j] = (
                fisher_metric(z + ei + ej, M, eps)
                - fisher_metric(z + ei - ej, M, eps)
                - fisher_metric(z - ei + ej, M, eps)
                + fisher_metric(z - ei - ej, M, eps)
            ) / (4 * h * h)
    return g, G1, G2
