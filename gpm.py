import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gpm_bases(model, dataloader, device, threshold=0.95, task_id=0):
    model.eval()
    handles = []
    cov_matrices = {}

    def get_hook(name, module):
        def hook(mod, inputs, outputs):
            x = inputs[0].detach()
            if isinstance(mod, nn.Conv2d):
                patches = F.unfold(
                    x, kernel_size=mod.kernel_size,
                    stride=mod.stride, padding=mod.padding
                )
                patches = patches.permute(1, 0, 2).reshape(patches.size(1), -1)
                cov = torch.matmul(patches, patches.t())
            else:
                return

            if name not in cov_matrices:
                cov_matrices[name] = cov
            else:
                cov_matrices[name] += cov
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(get_hook(name, module)))

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device, non_blocking=True)
            _ = model(x, task_id=task_id)

    for h in handles:
        h.remove()

    gpm_bases = {}
    for name, cov in cov_matrices.items():
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        cum_var = torch.cumsum(eigenvalues, dim=0) / total_var
        k = torch.searchsorted(cum_var, threshold).item() + 1
        k = max(1, min(k, eigenvectors.size(1)))

        gpm_bases[name] = eigenvectors[:, :k].contiguous()

    return gpm_bases

def project_backbone_gradients(model, gpm_bases):
    for name, module in model.named_modules():
        if name in gpm_bases and module.weight.grad is not None:
            M = gpm_bases[name]
            grad = module.weight.grad.data
            orig_shape = grad.shape

            grad_flat = grad.reshape(orig_shape[0], -1)
            proj = torch.matmul(torch.matmul(grad_flat, M), M.t())
            grad_ortho = grad_flat - proj

            module.weight.grad.data.copy_(grad_ortho.reshape(orig_shape))