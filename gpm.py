import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gpm_bases(model, dataloader, device, threshold=0.95, task_id=0,
                  max_samples=None, return_stats=False):
    """Salient-space bases for each Conv2d layer (Sharmin 2022, Sec. 4.3.2).

    Representation matrix R_l has input patch vectors of dimension
    C_in * k * k as its *columns* (eq. 4.5: grad W = X^T Delta, where the
    columns of X^T are the patch vectors).

    Autocorrelation  A_l = R_l R_l^T.  Since R_l = U S V^T, eigh(A_l) yields
    eigenvalues sigma_i^2 and eigenvectors U -- so the paper's energy criterion

        k = min{ k : ||X_k||_F^2 >= l_th * ||X||_F^2 }        (Sec. 4.3.2)
          = min{ k : sum_{i<=k} sigma_i^2 / sum_i sigma_i^2 >= l_th }   (eq. 4.6)

    is exactly a cumulative sum over the eigenvalues -- no square root involved.

    max_samples caps the images used for extraction. GPM (Saha et al.), which
    this chapter builds on, constructs R from ~1e2 samples; using the whole
    split flattens the spectrum and inflates k for a given l_th, tightening the
    constraint. Left as None to preserve existing behaviour by default.

    Returns bases, or (bases, stats) where stats[name] = (k, n) for the
    k/n vs l_th plot of Fig. 4.8.
    """
    was_training = model.training
    model.eval()
    handles, cov_matrices, seen = [], {}, 0

    def get_hook(name):
        def hook(mod, inputs, outputs):
            # A grouped conv's gradient does not decompose over a single dense
            # patch matrix; it would need one basis per group.
            if mod.groups != 1:
                return
            x = inputs[0].detach()
            patches = F.unfold(
                x, kernel_size=mod.kernel_size, dilation=mod.dilation,
                padding=mod.padding, stride=mod.stride,
            )                                                  # (N, C_in*k*k, L)
            D = patches.size(1)
            patches = patches.permute(1, 0, 2).reshape(D, -1)   # (D, N*L)
            cov = patches @ patches.t()
            # Column-concatenating batches means the Gram matrices simply add:
            # [R_1|...|R_B][R_1|...|R_B]^T = sum_b R_b R_b^T
            cov_matrices[name] = cov if name not in cov_matrices \
                else cov_matrices[name] + cov
        return hook

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device, non_blocking=True)
                _ = model(x, task_id=task_id)
                seen += x.size(0)
                if max_samples is not None and seen >= max_samples:
                    break
    finally:
        # Must survive an exception mid-extraction: leaked hooks would keep
        # accumulating covariance on every subsequent forward pass.
        for h in handles:
            h.remove()

    gpm_bases, stats = {}, {}
    for name in sorted(cov_matrices.keys()):
        # fp64 for the decomposition only: fp32 eigh resolves eigenvalues just
        # down to ~1e-7 * lambda_max and loses orthonormality of eigenvectors
        # for clustered eigenvalues -- and the whole method rests on M^T M = I.
        # pop() so the fp32 copy is freed before the fp64 one is built.
        cov = cov_matrices.pop(name).double()
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        del cov

        # eigh on a PSD matrix returns small negative eigenvalues in the
        # numerical null space; left in place they push cumsum above 1.0 and
        # back down, making the sequence non-monotone and searchsorted's
        # contract undefined.
        eigenvalues = eigenvalues.clamp_min(0)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        cum_var = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
        # searchsorted(side='left') gives the first index with cum_var >= l_th;
        # +1 converts that index into a component count.
        k = int(torch.searchsorted(cum_var, threshold).item()) + 1
        k = max(1, min(k, eigenvectors.size(1)))

        gpm_bases[name] = eigenvectors[:, :k].contiguous().float()
        stats[name] = (k, eigenvectors.size(1))

    if was_training:
        model.train()

    return (gpm_bases, stats) if return_stats else gpm_bases


@torch.no_grad()
def project_backbone_gradients(model, gpm_bases, return_norms=False):
    """Project backbone gradients into the null space of the salient space.

    Sec. 4.3.3 states  grad W_ortho = grad W - M M^T grad W,  with grad W
    shaped (C_in*k*k, C_out) by eq. 4.5. PyTorch stores the weight as
    (C_out, C_in*k*k) -- the paper's transpose -- so the equivalent form is

        G' = G - G M M^T,   G = weight.grad.reshape(C_out, C_in*k*k)

    which satisfies G' M = 0 since M^T M = I. The unfold flatten order
    (C_in, k_h, k_w) matches the weight flatten order, so the basis and the
    gradient rows live in the same coordinate system.

    Evaluated as (G M) M^T: O(C_out * D * k) and never forms the D x D projector.

    return_norms yields {name: (||G||, ||G'||)} for the gradient norm plots of
    Figs. 4.5 / 4.7.
    """
    norms = {}
    for name, module in model.named_modules():
        if name not in gpm_bases:
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.grad is None:
            continue

        M = gpm_bases[name]
        shape = weight.grad.shape
        flat = weight.grad.reshape(shape[0], -1)
        before = flat.norm().item() if return_norms else None

        ortho = flat - (flat @ M) @ M.t()
        weight.grad.copy_(ortho.reshape(shape))

        if return_norms:
            norms[name] = (before, ortho.norm().item())

    return norms if return_norms else None
