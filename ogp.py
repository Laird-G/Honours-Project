"""Orthogonal Gradient Projection for the robustness-accuracy trade-off.

Vision port of OGPSA (Sun et al., arXiv 2602.07892v2, "Safety Alignment as
Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient
Projection"), extended past the paper's Algorithm 1 to fix the failure modes
that only appear when the "general capability" being protected is clean image
accuracy rather than LLM general ability.

Paper baseline (Algorithm 1, p.18) -- reproduced by
  --ogp_ref ce --ogp_project equality --ogp_granularity global --ogp_alpha 1
    U <- GramSchmidt({grad L_ref_i}) every K steps
    g' <- g - U U^T g
    theta <- theta - eta g'

What this module adds, and why each one is needed here but not in the paper:

1. KL-ANCHOR REFERENCE OBJECTIVE (--ogp_ref kl / both).
   A converged WRN-28-10 has ~0 loss on its training data, so grad L_clean is
   noise-dominated; Gram-Schmidt normalises noise into a near-random unit vector
   whose cosine with the adversarial gradient is ~1/sqrt(d) ~ 1.6e-4 -- no
   constraint at all. LLM reference losses never vanish like that. The KL
   gradient, KL(f_theta(x) || f_theta0(x)) against a frozen copy of the
   pre-alignment model, is exactly the direction that undoes clean-function
   drift: zero at theta_0 (so the guard drops it and nothing is constrained
   before drift starts) and growing precisely as drift appears. It also needs no
   labels. This is the *hard* version of the KL anchoring the paper dismisses as
   a soft constraint (p.3).

2. GEM-STYLE INEQUALITY CONSTRAINT (--ogp_project inequality).
   Eq. 12 enforces <g_ref, delta theta> = 0, i.e. it *preserves* the reference
   loss. For a trade-off we only need it not to get WORSE: <g_ref, g'> >= 0. The
   equality version therefore discards the shared descent direction even when
   both objectives agree, paying robustness for no protection. The inequality
   version solves the exact GEM dual QP (Lopez-Paz & Ranzato) rather than
   PCGrad's sequential heuristic, and costs essentially nothing extra -- see
   _nnqp and the R-factor trick below. It strictly generalises eq. 12: when
   every constraint is active the solution IS the orthogonal projection.

3. GRANULARITY (--ogp_granularity global|per_tensor).
   The paper projects the whole flattened parameter vector, removing M'
   directions out of d ~ 3.65e7. Per-tensor projection imposes M' constraints
   *per tensor* -- a strictly stronger constraint, and the rung that moves this
   method toward GPM's per-layer structure if the global version proves too weak.

4. ALPHA INTERPOLATION (--ogp_alpha).
   g' = g - alpha * (correction). alpha=0 leaves training unconstrained but
   still computes and logs the reference gradients and conflict statistics, so
   the control run reports how much conflict the projection *would* have
   removed. Sweeping alpha traces a frontier, which is what a
   "mitigates the trade-off" claim needs -- the claim is about curves, not points.

5. RENORMALISATION (--ogp_renorm).
   The projection shrinks ||g'||, so "projection helped" can just mean "smaller
   steps". Rescaling g' to ||g|| removes that confound, and is arguably closer
   to the paper's own theory than eq. 13 is: Prop. 4.1 / eq. 16 characterises
   the *unit* steepest feasible direction.

Implementation notes that matter for correctness:

* Everything works on lists of parameter-shaped tensors, never one flat 146 MB
  cat. An inner product is the sum of per-tensor dots, which is exactly the flat
  inner product.
* Reductions are sync-frugal. Every device-side dot needed by one projection is
  issued without synchronising, stacked, and copied to host ONCE. A naive
  implementation costs ~110 syncs per dot and would dominate the step.
* The R-factor trick: Gram-Schmidt records the coefficients expressing each
  normalised reference direction a_i in the orthonormal basis, a_i = sum_j
  R[i,j] u_j. Then <a_i, g> = (R c)_i where c_j = <u_j, g> are the M' inner
  products the equality projection already computes, and the GEM Gram matrix is
  W = R R^T -- free. So the inequality mode needs no extra GPU reductions over
  the equality mode: M' dots, one sync, a tiny CPU QP, M' axpys.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# A reference direction is discarded when its norm is below this fraction of the
# largest reference norm in the same refresh. Catches the KL objective at
# theta = theta_0 (exactly zero) and any direction that has collapsed to noise.
MIN_REL_NORM = 1e-3


# ---------------------------------------------------------------------------
# Flat-vector primitives over lists of parameter-shaped tensors
# ---------------------------------------------------------------------------

def _dot_dev(a, b):
    """<a, b> as a 0-dim float64 DEVICE tensor. Issues no host sync."""
    total = None
    for at, bt in zip(a, b):
        d = torch.dot(at.reshape(-1), bt.reshape(-1)).double()
        total = d if total is None else total + d
    return total


def flat_dot(a, b):
    """<a, b> as a Python float. One host sync."""
    t = _dot_dev(a, b)
    return 0.0 if t is None else float(t)


def flat_norm(a):
    return math.sqrt(max(0.0, flat_dot(a, a)))


# ---------------------------------------------------------------------------
# Reference data: pools disjoint from the selection split
# ---------------------------------------------------------------------------

class _CyclingLoader:
    """Endless sampler over one reference pool (Algorithm 1 line 5).

    D_ref_i is a *fixed* dataset; B_i is a fresh mini-batch drawn from it at
    every refresh, so the iterator restarts rather than exhausting.
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


def split_reference_and_selection(valloader, num_refs, ref_samples, ref_batch,
                                  eval_batch_size, num_workers=0, seed=1234):
    """Carve the validation split into reference pools + a disjoint selection set.

    Three-way separation, and all three parts matter:

      * training data      -- adversarial fine-tuning. Excluded from reference
        use because a converged model has ~0 loss there, so its clean gradient
        is noise (see the KL discussion in the module docstring).
      * reference pools    -- held out, so L_ref is genuinely non-zero.
      * selection set      -- held out AND disjoint from the reference pools, so
        checkpoint selection is not scored on data that steered the projection.

    Returns (ref_loaders, selection_loader). Reference pools are mutually
    disjoint, matching the paper's M separate reference datasets.

    num_workers = 0 for the pools: they hold a few hundred images, so worker
    startup on every refresh would cost more than the load itself.
    """
    dataset = valloader.dataset
    needed = num_refs * ref_samples
    if needed >= len(dataset):
        raise ValueError(
            f"Reference pools need {num_refs} x {ref_samples} = {needed} images but the "
            f"validation split only has {len(dataset)}, and some must be left for "
            f"checkpoint selection. Reduce --ogp_num_refs or --ogp_ref_samples."
        )
    if num_refs > 0 and ref_batch > ref_samples:
        raise ValueError(f"ref_batch ({ref_batch}) cannot exceed ref_samples ({ref_samples}).")

    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()

    ref_loaders = []
    for i in range(num_refs):
        idx = perm[i * ref_samples:(i + 1) * ref_samples]
        ref_loaders.append(_CyclingLoader(DataLoader(
            Subset(dataset, idx), batch_size=ref_batch, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )))

    selection_loader = DataLoader(
        Subset(dataset, perm[needed:]), batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return ref_loaders, selection_loader


# ---------------------------------------------------------------------------
# Reference gradients (Algorithm 1 lines 4-7)
# ---------------------------------------------------------------------------

def _snapshot(params):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in params]


def reference_gradients(model, ref_loaders, params, device, task_id=0,
                        objectives=("ce", "kl"), temperature=2.0, criterion=None,
                        teacher=None, per_class=False, num_classes=10):
    """Reference gradients at the current theta. Returns (grads, labels).

    grads[i] is a list of tensors aligned with `params`; labels[i] names it for
    the diagnostics.

    The model is put in eval mode for these passes, and that is a correctness
    requirement rather than a stylistic choice: BatchNorm in train mode mutates
    running_mean / running_var inside forward(), so a train-mode reference pass
    would silently change the model -- a side effect Algorithm 1 does not have.
    eval mode also makes L_ref the loss of the *deployed* function, the same
    convention attacks.pgd_attack uses. Per-module save/restore so it composes
    with a selectively frozen backbone.

    Computed in fp32 outside autocast and without a GradScaler. The projector is
    scale-equivariant and every direction is normalised, so a loss scale would
    cancel; fp32 removes any chance of the reference *direction* being corrupted
    by fp16 underflow, at the cost of one extra fp32 forward/backward per pool
    per refresh (<1% of runtime at K=30).

    temperature > 1 softens the logits. For `ce` that keeps the gradient from
    collapsing toward zero on well-fit data; for `kl` it is the standard
    distillation temperature. Only the direction survives normalisation, so the
    usual tau^2 rescaling is irrelevant here.
    """
    if criterion is None:
        criterion = F.cross_entropy
    if "kl" in objectives and teacher is None:
        raise ValueError("objectives include 'kl' but no frozen teacher model was given.")

    prev_modes = {m: m.training for m in model.modules()}
    model.eval()
    grads, labels = [], []
    try:
        with torch.enable_grad():
            for pool_i, loader in enumerate(ref_loaders):
                x, y = loader.next_batch()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # One forward serves every objective on this batch; the graph is
                # retained across the per-objective backwards.
                logits = model(x, task_id=task_id) / temperature
                teacher_logits = None
                if "kl" in objectives:
                    with torch.no_grad():
                        teacher_logits = teacher(x, task_id=task_id) / temperature

                losses = []
                if per_class:
                    # Genuinely distinct facets: one direction per class, all
                    # from a single mixed-batch forward so BatchNorm statistics
                    # are not distorted by single-class batches (they would be
                    # in eval mode anyway, but the activations still differ).
                    for c in range(num_classes):
                        m = (y == c)
                        if m.sum() == 0:
                            continue
                        if "ce" in objectives:
                            losses.append((f"ce/c{c}", criterion(logits[m], y[m])))
                        if "kl" in objectives:
                            losses.append((f"kl/c{c}", F.kl_div(
                                F.log_softmax(logits[m], dim=1),
                                F.softmax(teacher_logits[m], dim=1),
                                reduction="batchmean")))
                else:
                    if "ce" in objectives:
                        losses.append((f"ce/p{pool_i}", criterion(logits, y)))
                    if "kl" in objectives:
                        losses.append((f"kl/p{pool_i}", F.kl_div(
                            F.log_softmax(logits, dim=1),
                            F.softmax(teacher_logits, dim=1),
                            reduction="batchmean")))

                for n, (label, loss) in enumerate(losses):
                    model.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=(n < len(losses) - 1))
                    grads.append(_snapshot(params))
                    labels.append(label)

                del logits, teacher_logits, losses
                if per_class:
                    break        # per_class draws all its facets from one batch
    finally:
        # A reference backward must never leak into the adversarial gradient.
        model.zero_grad(set_to_none=True)
        for m, mode in prev_modes.items():
            m.training = mode
    return grads, labels


# ---------------------------------------------------------------------------
# The subspace: Gram-Schmidt with R factor, projection, GEM dual QP
# ---------------------------------------------------------------------------

def _nnqp(W, b, step, iters):
    """min over v >= 0 of  0.5 v^T W v + v^T b   (the GEM dual).

    Projected gradient descent. W is M x M with M <= ~20, so this is
    microseconds on CPU and needs no external solver. W is PSD and the feasible
    set is a box, so the iteration is monotone; the caller feasibility-checks the
    result and falls back to the always-feasible orthogonal projection if the
    iteration has not converged far enough.
    """
    v = torch.zeros_like(b)
    for _ in range(iters):
        v = torch.clamp(v - step * (W @ v + b), min=0.0)
    return v


def _gram_schmidt_group(vectors, delta):
    """Eq. 10-11 over one parameter group, also returning the R factor.

    Each input is normalised to unit length BEFORE orthogonalisation, so the
    residual ||v_k|| lands in [0, 1] and `delta` is a dimensionless "fraction of
    this direction that is new" threshold. Eq. 11 as written compares ||v_k||
    against a delta in gradient units and the paper never states its value;
    since the projector is invariant to positive rescaling of the basis,
    pre-normalising changes nothing but makes delta portable.

    Returns (basis, R, residuals) with  a_i = sum_j R[i,j] u_j  where
    a_i = vectors[i] / ||vectors[i]||, exact up to a discarded residual < delta.
    """
    basis, rows, residuals = [], [], []
    for g in vectors:
        n0 = flat_norm(g)
        if n0 <= 1e-12:
            rows.append([])
            residuals.append(0.0)
            continue

        v = [t / n0 for t in g]
        coeffs = []
        for u in basis:
            c = flat_dot(v, u)
            coeffs.append(c)
            for vt, ut in zip(v, u):
                vt.sub_(ut, alpha=c)

        nv = flat_norm(v)
        residuals.append(nv)
        if nv >= delta:
            for vt in v:
                vt.div_(nv)
            basis.append(v)
            coeffs.append(nv)
        rows.append(coeffs)

    mp = len(basis)
    R = torch.zeros(len(vectors), mp, dtype=torch.float64)
    for i, row in enumerate(rows):
        for j, c in enumerate(row[:mp]):
            R[i, j] = c
    return basis, R, residuals


class _Group:
    __slots__ = ("idx", "basis", "R", "W", "qp_step")


class ReferenceSubspace:
    """The protected subspace and the projection rule built on it.

    mode="equality"   -- eq. 12, the paper's rule: remove the whole component.
    mode="inequality" -- GEM: remove only what would raise the reference loss.
    """

    def __init__(self, params, granularity="global", delta=0.05, mode="equality",
                 alpha=1.0, renorm=False, qp_iters=300, feas_tol=1e-6,
                 min_rel_norm=MIN_REL_NORM):
        if granularity == "global":
            self._group_idx = [list(range(len(params)))]
        elif granularity == "per_tensor":
            self._group_idx = [[i] for i in range(len(params))]
        else:
            raise ValueError(f"unknown granularity {granularity!r}")
        if mode not in ("equality", "inequality"):
            raise ValueError(f"unknown projection mode {mode!r}")

        self.granularity = granularity
        self.delta = delta
        self.mode = mode
        self.alpha = alpha
        self.renorm = renorm
        self.qp_iters = qp_iters
        self.feas_tol = feas_tol
        self.min_rel_norm = min_rel_norm

        self.groups = []
        self.labels = []
        self.residuals = []
        self.ref_norms = []
        self.dropped_weak = 0

    # -- construction -------------------------------------------------------

    def build(self, grads, labels=None):
        """Rebuild from a fresh set of reference gradients. Returns retained rank."""
        labels = labels or [f"g{i}" for i in range(len(grads))]
        self.ref_norms = [flat_norm(g) for g in grads]
        if not self.ref_norms or not all(math.isfinite(n) for n in self.ref_norms):
            return None                      # caller keeps the previous basis

        # Drop directions that have collapsed relative to the strongest one --
        # notably the KL objective at theta = theta_0, which is exactly zero.
        floor = self.min_rel_norm * max(self.ref_norms)
        keep = [i for i, n in enumerate(self.ref_norms) if n > floor]
        self.dropped_weak = len(grads) - len(keep)
        self.labels = [labels[i] for i in keep]
        kept = [grads[i] for i in keep]

        groups, residuals = [], []
        for idx in self._group_idx:
            sliced = [[g[i] for i in idx] for g in kept]
            basis, R, resid = _gram_schmidt_group(sliced, self.delta)
            grp = _Group()
            grp.idx, grp.basis, grp.R = idx, basis, R
            grp.W = R @ R.transpose(0, 1)
            if len(basis) and self.mode == "inequality":
                lam = float(torch.linalg.eigvalsh(grp.W).max().clamp_min(1e-12))
                grp.qp_step = 1.0 / lam
            else:
                grp.qp_step = 0.0
            groups.append(grp)
            residuals.append(resid)

        self.groups = groups
        # For the global case there is one group; per_tensor reports group 0.
        self.residuals = residuals[0] if residuals else []
        return self.rank

    @property
    def rank(self):
        return max((len(g.basis) for g in self.groups), default=0)

    @property
    def active(self):
        return self.rank > 0 and self.alpha != 0.0

    # -- the projection itself ----------------------------------------------

    @torch.no_grad()
    def project(self, params, return_stats=False):
        """g' = g - alpha*U U^T g  (equality)  or  g + alpha*sum v_i a_i  (GEM).

        In place on p.grad. Returns a stats dict when asked, else None.

        Every device-side reduction is issued before the single host sync, so
        the cost is one sync per call regardless of granularity or rank.
        """
        if not self.groups:
            return None
        for i, p in enumerate(params):
            if p.grad is None:
                raise RuntimeError(
                    f"params[{i}] (shape {tuple(p.shape)}) has no gradient at projection "
                    f"time; the parameter list must match the tensors that receive gradients."
                )

        need_norm = return_stats or self.renorm

        # ---- phase 1: queue every reduction, no sync ----
        scalars, plan = [], []
        for grp in self.groups:
            g = [params[i].grad for i in grp.idx]
            gn2_pos = None
            if need_norm:
                gn2_pos = len(scalars)
                scalars.append(_dot_dev(g, g))
            c_start = len(scalars)
            for u in grp.basis:
                scalars.append(_dot_dev(g, u))
            plan.append((grp, g, gn2_pos, c_start, len(grp.basis)))

        if not scalars:
            return None
        vals = torch.stack(scalars).cpu()          # the one and only sync

        # ---- phase 2: tiny CPU maths, then apply ----
        gn2, corr, c_sq = 0.0, 0.0, 0.0
        cosines, n_active, n_fallback = [], 0, 0

        for grp, g, gn2_pos, c_start, nb in plan:
            if gn2_pos is not None:
                gn2 += float(vals[gn2_pos])
            if nb == 0:
                continue
            c = vals[c_start:c_start + nb].clone()

            if self.mode == "equality":
                w = -self.alpha * c
            else:
                # GEM dual: b_i = <a_i, g> = (R c)_i, Gram matrix W = R R^T.
                b = grp.R @ c
                v = _nnqp(grp.W, b, grp.qp_step, self.qp_iters)
                margin = b + grp.W @ v          # = <a_i, g'> before alpha scaling
                tol = self.feas_tol * max(1.0, float(b.abs().max()))
                if float(margin.min()) < -tol:
                    # The orthogonal projection is always feasible, so it is the
                    # safe fallback when the dual iteration has not converged.
                    w = -self.alpha * c
                    n_fallback += 1
                else:
                    w = self.alpha * (grp.R.transpose(0, 1) @ v)
                    n_active += int((v > 0).sum())

            for j, u in enumerate(grp.basis):
                wj = float(w[j])
                if wj == 0.0:
                    continue
                for gt, ut in zip(g, u):
                    gt.add_(ut, alpha=wj)

            if need_norm:
                # The basis is orthonormal, so ||g'||^2 = ||g||^2 + 2<w,c> + <w,w>
                # exactly -- no second reduction pass needed.
                corr += float((2.0 * w * c + w * w).sum())
                c_sq += float((c * c).sum())
            if return_stats and self.granularity == "global":
                cosines = [float(x) for x in c]

        if not need_norm:
            return None

        before = math.sqrt(max(0.0, gn2))
        after = math.sqrt(max(0.0, gn2 + corr))

        if self.renorm and after > 0 and before > 0:
            # Prop. 4.1 / eq. 16 is stated for the *unit* steepest feasible
            # direction, so restoring the norm keeps the step size out of the
            # comparison between projected and unprojected runs.
            scale = before / after
            for grp, g, _, _, _ in plan:
                for gt in g:
                    gt.mul_(scale)
            after = before

        return {
            "before": before,
            "after": after,
            "ratio": (after / before) if before > 0 else float("nan"),
            "frac_in_span": (math.sqrt(max(0.0, c_sq)) / before) if before > 0 else float("nan"),
            "cosines": [c / before for c in cosines] if before > 0 else [],
            "n_active": n_active,
            "n_fallback": n_fallback,
        }

    # -- correctness check --------------------------------------------------

    @torch.no_grad()
    def selfcheck(self, params, tol=1e-3, scale=None):
        """Verify the basis is orthonormal and the projection did what it claims.

        Run once, on the first projected step. The bugs this exists to catch -- a
        mismatched flatten order, a stale basis, the wrong parameter list --
        leave errors of order 0.1 to 1, whereas a correct fp32 reduction over
        ~3.7e7 elements still accumulates ~1e-6. Hence a loose tolerance: it
        costs no detection power and avoids aborting a long run over noise.

        equality   -> g' must be orthogonal to every basis vector.
        inequality -> g' must NOT be (that is the point); instead every
                      reference direction must satisfy <a_i, g'> >= 0, which is
                      the constraint the QP was solving.

        `scale` is the PRE-projection ||g||, and passing it matters. Normalising
        the residuals by the post-projection ||g'|| blows up whenever the
        projection legitimately annihilates the gradient (g lying entirely in the
        protected span): the residuals are then float noise over ~0, which reads
        as a gross violation when in fact g' = 0 satisfies every constraint
        trivially. ||g|| is the stable yardstick and never degenerates.
        """
        gram_err, resid, feas = 0.0, 0.0, float("inf")
        for grp in self.groups:
            for i, ui in enumerate(grp.basis):
                for j, uj in enumerate(grp.basis):
                    gram_err = max(gram_err, abs(flat_dot(ui, uj) - (1.0 if i == j else 0.0)))
            if not grp.basis:
                continue
            g = [params[k].grad for k in grp.idx]
            norm = flat_norm(g)
            denom = scale if (scale is not None and scale > 0) else norm
            if denom <= 0:
                continue
            c = torch.tensor([flat_dot(g, u) for u in grp.basis], dtype=torch.float64)
            resid = max(resid, float(c.abs().max()) / denom)
            feas = min(feas, float((grp.R @ c).min()) / denom)

        if self.mode == "equality":
            ok = gram_err < tol and resid < tol
            detail = f"max|<u_j, g'>|/||g'|| = {resid:.2e}"
        else:
            ok = gram_err < tol and feas > -tol
            detail = f"min <a_i, g'>/||g'|| = {feas:+.2e} (want >= 0)"
        return {"gram_err": gram_err, "resid": resid, "feas": feas,
                "ok": ok, "detail": detail}
