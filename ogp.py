"""Orthogonal Gradient Projection (OGPSA), Sun et al., arXiv 2602.07892v2.

Faithful port of Algorithm 1 (p.18):

    U <- []
    for t = 0 .. T-1:
        if t mod K == 0:                          # dynamic subspace refresh
            for i = 1 .. M:
                B_i ~ D_ref_i
                g_i <- grad_theta L_ref_i(theta_t; B_i)
            U <- GramSchmidt({g_i})               # eq. 10-11, delta-filtered
        g  <- grad_theta L_safe(theta_t)
        g' <- g - U (U^T g)                       # eq. 12
        theta <- theta - eta g'                   # eq. 13, no renormalisation

The subspace lives in *parameter* space: the whole trainable parameter vector is
treated as one d-dimensional vector (d ~ 3.65e7 for WRN-28-10), which is what
eq. 12 means by U^T g. That is the difference from gpm.py, where the basis lives
in each layer's input-patch space and the projection is applied per layer.

Everything here operates on lists of parameter-shaped tensors rather than one
flat vector, so no 146 MB torch.cat is ever materialised; inner products are the
sum of per-tensor dot products, which is exactly the flat inner product.
"""

import math

import torch
from torch.utils.data import DataLoader, Subset


def flat_dot(a, b):
    """<a, b> for two lists of parameter-shaped tensors.

    Per-tensor dots in fp32 (PyTorch reduces pairwise, so each is accurate),
    accumulated in fp64 -- but as a 0-dim *device* tensor, so the whole reduction
    costs one host-device sync instead of one per tensor. That matters: this runs
    on every training step, and ~110 syncs per call would cost more than the
    projection itself.
    """
    total = None
    for at, bt in zip(a, b):
        d = torch.dot(at.reshape(-1), bt.reshape(-1)).double()
        total = d if total is None else total + d
    return 0.0 if total is None else float(total)


def flat_norm(a):
    return math.sqrt(max(0.0, flat_dot(a, a)))


class _CyclingLoader:
    """Endless sampler over one reference pool (Algorithm 1 line 5).

    D_ref_i is a *fixed* dataset; B_i is a fresh mini-batch drawn from it at
    every refresh, so the iterator is restarted rather than exhausted.
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


def make_reference_loaders(base_loader, num_refs, ref_samples, ref_batch, seed=1234):
    """Partition base_loader's dataset into M disjoint fixed reference pools.

    The paper uses M = 2 reference sets of 200 samples each, drawn from corpora
    held out from the alignment data (Appendix A). Here they come from the
    un-augmented validation view built in dataset.get_dataloaders -- the same
    clean, deterministic source gpm.get_gpm_bases extracts its basis from.

    num_workers = 0: the pools are ~200 images, so worker startup on every
    refresh would cost more than the load itself.
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


def reference_gradients(model, ref_loaders, criterion, params, device, task_id=0):
    """Algorithm 1 lines 4-7: one reference gradient per pool at the current theta.

    Computed in fp32 outside autocast and without a GradScaler. The projector is
    scale-equivariant and Gram-Schmidt normalises, so a loss scale would cancel
    anyway; running in fp32 removes any chance of the reference *direction* being
    corrupted by fp16 underflow, at the cost of one extra fp32 forward/backward
    per pool every K steps.

    The model is put in eval mode for these passes. That is not a fidelity
    choice: BatchNorm in train mode mutates running_mean/running_var inside
    forward(), so a train-mode reference pass would silently change the model --
    a side effect Algorithm 1 does not have. eval mode also makes L_ref the loss
    of the *deployed* function, the same convention attacks.pgd_attack uses.
    Per-module save/restore rather than train()/eval() so it composes with any
    selectively frozen submodule.
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
                loss = criterion(model(x, task_id=task_id), y)
                loss.backward()

                grads.append([
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in params
                ])
    finally:
        # The reference backward must not leak into the adversarial gradient.
        model.zero_grad(set_to_none=True)
        for m, mode in prev_modes.items():
            m.training = mode
    return grads


def gram_schmidt(grads, delta=0.1, eps=1e-12):
    """Eq. 10-11. Returns (basis, residuals).

    Each reference gradient is normalised to unit length *before*
    orthogonalisation, so the residual norm ||v_k|| lands in [0, 1] and delta is
    a dimensionless "fraction of this direction that is new" threshold. Eq. 11
    as written compares ||v_k|| against a delta in the units of the gradient,
    and the paper never states its value; since the projector is invariant to a
    positive rescaling of the basis vectors, pre-normalising changes nothing but
    makes delta portable.

    residuals[i] is ||v_i|| for each input direction: 0.0 means the reference
    gradient was numerically zero (a dead reference objective), and a value
    below delta means it was near-collinear with an earlier one and was dropped.
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
    """Eq. 12: g' = g - U (U^T g), in place on p.grad.

    Mirrors gpm.project_backbone_gradients, but over the whole parameter vector
    instead of per layer.

    <g, u_j> is read before subtracting u_j's component and after subtracting
    u_1..u_{j-1}'s. Because the basis is orthonormal those earlier subtractions
    leave <g, u_j> unchanged, so each coefficient equals <g_original, u_j> and
    the loop is exact, not a Gauss-Seidel approximation.

    return_stats yields (||g||, ||g'||, [cos(g, u_j)]) for the gradient-conflict
    diagnostics, in the spirit of gpm.project_backbone_gradients' return_norms.
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
def selfcheck(basis, params, tol=1e-4):
    """One-off correctness check, run at the first projected step.

    Verifies (a) the basis is orthonormal and (b) the projected gradient is
    actually orthogonal to it. Both catch the flatten/ordering class of bug that
    would otherwise produce plausible-looking curves from a wrong projection.
    Returns (max |U^T U - I|, max |<u_j, g'>| / ||g'||, passed).

    tol is 1e-4, not machine precision: these are fp32 reductions over ~3.7e7
    elements, so a correct implementation still accumulates error around 1e-6.
    The bugs this is here to catch -- a mismatched flatten order, a stale basis,
    a wrong parameter list -- leave residuals of order 0.1 to 1, so a looser
    tolerance costs no detection power and avoids aborting a long run over
    numerical noise.
    """
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
