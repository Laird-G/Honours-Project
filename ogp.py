"""Orthogonal Gradient Projection (OGPSA), Sun et al., arXiv 2602.07892v2[cite: 1].

Enhanced Vision Port addressing key vision-adaptation failure modes:
1. BatchNorm drift is addressed in trainer.py by freezing statistics[cite: 3].
2. Vanishing reference gradients at clean convergence are resolved via temperature scaling[cite: 3].
3. Subspace expressiveness is improved by default to multi-pool reference loaders[cite: 1, 2].
"""

import math
import torch
from torch.utils.data import DataLoader, Subset


def flat_dot(a, b):
    """<a, b> for two lists of parameter-shaped tensors[cite: 2].

    Per-tensor dots in fp32, accumulated in fp64 as a 0-dim device tensor[cite: 2].
    This costs one host-device sync across all parameter tensors rather than
    one sync per tensor on every training iteration[cite: 2].
    """
    total = None
    for at, bt in zip(a, b):
        d = torch.dot(at.reshape(-1), bt.reshape(-1)).double()
        total = d if total is None else total + d
    return 0.0 if total is None else float(total)


def flat_norm(a):
    """Euclidean norm of a list of parameter-shaped tensors[cite: 2]."""
    return math.sqrt(max(0.0, flat_dot(a, a)))


class _CyclingLoader:
    """Endless sampler over one reference pool (Algorithm 1 line 5)[cite: 1, 2].

    D_ref_i is a fixed dataset; B_i is a fresh mini-batch drawn from it at
    every refresh, so the iterator is restarted rather than exhausted[cite: 1, 2].
    """

    def __init__(self, loader):
        self.loader = loader
        self._it = None

    def next_batch(self):
        if self._it is None:
            self._it = iter(self.loader)
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def make_reference_loaders(base_loader, num_refs=8, ref_samples=256, ref_batch=64, seed=1234):
    """Partition base_loader's dataset into M disjoint fixed reference pools[cite: 2].

    Increasing num_refs (e.g., from M=2 to M=8 or 16) provides a richer subspace
    basis spanning diverse class manifolds rather than stochastic mini-batch noise[cite: 1, 2].
    num_workers = 0 prevents worker startup latency bottlenecks on periodic refreshes[cite: 2].
    """
    dataset = base_loader.dataset
    needed = num_refs * ref_samples
    if needed > len(dataset):
        raise ValueError(
            f"Need {num_refs} x {ref_samples} = {needed} reference images but the "
            f"pool source only has {len(dataset)}."
        )
    if ref_batch > ref_samples:
        raise ValueError(f"ref_batch ({ref_batch}) cannot exceed ref_samples ({ref_samples}).")

    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    loaders = []
    for i in range(num_refs):
        idx = perm[i * ref_samples:(i + 1) * ref_samples]
        loaders.append(_CyclingLoader(DataLoader(
            Subset(dataset, idx), batch_size=ref_batch, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )))
    return loaders


def reference_gradients(model, ref_loaders, criterion, params, device, task_id=0, temperature=2.0):
    """Algorithm 1 lines 4-7: Reference gradients computed with temperature scaling[cite: 1, 2].

    WHY TEMPERATURE SCALING MATTERS:
    At a pre-trained checkpoint, cross-entropy loss is near zero, meaning ||grad L_ref|| -> 0[cite: 3].
    Standard gradients reflect mini-batch noise rather than the clean manifold[cite: 3]. Dividing
    logits by tau > 1 softens probabilities, preventing zero-gradient collapse and producing
    strong, informative clean reference vectors[cite: 3].
    """
    prev_modes = {m: m.training for m in model.modules()}
    model.eval()
    grads = []
    try:
        with torch.enable_grad():
            for loader in ref_loaders:
                x, y = loader.next_batch()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                model.zero_grad(set_to_none=True)
                
                # Temperature scaling softens the distribution to produce non-vanishing gradients
                logits = model(x, task_id=task_id) / temperature
                loss = criterion(logits, y)
                loss.backward()

                grads.append([
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in params
                ])
    finally:
        # Prevent reference backward passes from leaking into the adversarial gradient
        model.zero_grad(set_to_none=True)
        for m, mode in prev_modes.items():
            m.training = mode
    return grads


def gram_schmidt(grads, delta=0.05, eps=1e-12):
    """Eq. 10-11: Gram-Schmidt orthonormalization with redundancy thresholding[cite: 1, 2].

    Each reference gradient is normalized to unit length before orthogonalization[cite: 2].
    Residuals below delta are discarded as collinear directions[cite: 1, 2].
    """
    basis, residuals = [], []
    for g in grads:
        n0 = flat_norm(g)
        if n0 <= eps:
            residuals.append(0.0)
            continue

        v = [t / n0 for t in g]
        for u in basis:
            c = flat_dot(v, u)
            for vt, ut in zip(v, u):
                vt.sub_(ut, alpha=c)

        nv = flat_norm(v)
        residuals.append(nv)
        if nv < delta:
            continue
        for vt in v:
            vt.div_(nv)
        basis.append(v)
    return basis, residuals


@torch.no_grad()
def project_orthogonal(basis, params, return_stats=False):
    """Eq. 12: Projects safety/adversarial gradient onto S_gen^perp[cite: 1, 2].

    g' = g - U (U^T g)[cite: 2]
    Applied in-place on p.grad across all parameter tensors[cite: 2].
    """
    for i, p in enumerate(params):
        if p.grad is None:
            raise RuntimeError(
                f"params[{i}] has no gradient at projection time; the parameter list "
                f"must match the tensors that receive gradients (shape {tuple(p.shape)})."
            )

    grads = [p.grad for p in params]
    before = flat_norm(grads) if return_stats else None

    cosines = []
    for u in basis:
        c = flat_dot(grads, u)
        if return_stats:
            cosines.append(c / before if before and before > 0 else 0.0)
        for gt, ut in zip(grads, u):
            gt.sub_(ut, alpha=c)

    if not return_stats:
        return None
    return before, flat_norm(grads), cosines


@torch.no_grad()
def selfcheck(basis, params, tol=1e-3):
    """Verifies basis orthonormality and gradient projection orthogonality[cite: 2]."""
    max_gram_err = 0.0
    for i, ui in enumerate(basis):
        for j, uj in enumerate(basis):
            expected = 1.0 if i == j else 0.0
            max_gram_err = max(max_gram_err, abs(flat_dot(ui, uj) - expected))

    grads = [p.grad for p in params]
    norm = flat_norm(grads)
    max_resid = 0.0
    if norm > 0:
        for u in basis:
            max_resid = max(max_resid, abs(flat_dot(grads, u)) / norm)

    return max_gram_err, max_resid, (max_gram_err < tol and max_resid < tol)