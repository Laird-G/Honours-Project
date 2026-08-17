import torch
import torch.nn.functional as F
from torch.amp import autocast

# PGD only ever uses grad.sign(), which is invariant to any positive rescaling
# of the loss. The attack runs under autocast with no GradScaler attached, so
# fp16 backward intermediates can flush to zero (fp16 min normal ~6.1e-5, min
# subnormal ~5.96e-8, while input gradients of a converged CIFAR net sit around
# 1e-6..1e-8). sign() then returns 0 and those pixels are silently left
# unperturbed, which weakens the attack and inflates robust accuracy.
# Scaling lifts them out of the subnormal band without changing the direction.
# 2^12 leaves ample headroom: d(CE)/d(logits) is bounded in [-1, 1], so the
# largest scaled intermediate is ~4096 against an fp16 max of 65504.
_ATTACK_LOSS_SCALE = 2.0 ** 12


def pgd_attack(model, images, targets, eps=8/255, alpha=2/255, steps=7,
               random_start=True, task_id=0):
    """L_inf PGD, Madry et al. 2019 Sec. 2.1:

        x_{t+1} = Proj_{x+S} ( x_t + alpha * sign(grad_x L(theta, x, y)) )

    Operates entirely in unnormalized [0, 1] pixel space; NormalizedModel
    applies the mean/std transform inside its forward pass, so eps is a true
    eps/255 perturbation on raw pixels.
    """
    # Generate against the deployed function (BN running statistics), and do
    # not let the attack's forward passes mutate running_mean / running_var.
    # Per-module save/restore rather than a train()/eval() pair so this
    # composes with a selectively frozen backbone (see trainer.freeze_backbone_bn).
    prev_modes = {m: m.training for m in model.modules()}
    model.eval()
    try:
        with torch.enable_grad():
            adv_images = images.clone().detach()
            if random_start:
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
                # Clamping can only move the point toward the clean image, so
                # the start stays inside the eps-ball.
                adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

            for _ in range(steps):
                adv_images.requires_grad_(True)
                with autocast('cuda'):
                    outputs = model(adv_images, task_id=task_id)
                    loss = F.cross_entropy(outputs, targets)

                grad = torch.autograd.grad(loss * _ATTACK_LOSS_SCALE, adv_images)[0]
                # A single non-finite element would poison sign() and propagate
                # through the clamps as NaN. nan->0, +-inf->+-max_float, so
                # sign() still behaves sensibly.
                grad = torch.nan_to_num(grad)
                adv_images = adv_images.detach() + alpha * grad.sign()

                # Project onto the eps-ball, then onto [0, 1]. Both sets are
                # coordinatewise boxes, so this composition is exactly the
                # projection onto their intersection, not an approximation.
                eta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()
    finally:
        for m, mode in prev_modes.items():
            m.training = mode

    return adv_images
