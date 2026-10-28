import torch

KEY = 0
V_MIN_NOISE = 16


def _gen(seed):
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
    return g


def sample_passkey(B, L, V, values=8, device="cpu", needle_min=8, needle_max=None, seed=None):
    g = _gen(seed)
    if needle_max is None:
        needle_max = L - 4
    noise_lo, noise_hi = V_MIN_NOISE, V
    seq = torch.randint(noise_lo, noise_hi, (B, L), generator=g)
    vals = torch.randint(1, values + 1, (B,), generator=g)
    pos = torch.randint(needle_min, needle_max, (B,), generator=g)
    for i in range(B):
        seq[i, pos[i]] = KEY
        seq[i, pos[i] + 1] = vals[i]
        seq[i, -1] = KEY
    return seq.to(device), vals.to(device), pos


def sample_copy(B, L, V, device="cpu", seed=None):
    g = _gen(seed)
    n = (L - 1) // 2
    half = torch.randint(V_MIN_NOISE, V, (B, n), generator=g)
    seq = torch.cat([half, torch.full((B, 1), KEY, dtype=torch.long), half], dim=1)
    return seq.to(device), half.to(device)


def passkey_accuracy(model, B, L, V, device, seed=None):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, B, 64):
            b = min(64, B - start)
            seq, vals, _ = sample_passkey(b, L, V, device=device, seed=None if seed is None else seed + start)
            logits = model(seq)
            pred = logits[:, -1].argmax(-1)
            correct += (pred == vals).sum().item()
    return correct / B
