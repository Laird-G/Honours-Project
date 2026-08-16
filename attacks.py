import torch
import torch.nn.functional as F

def pgd_attack(model, images, targets, eps=8/255, alpha=2/255, steps=7, random_start=True):
    """
    Projected Gradient Descent (PGD) L_infinity adversary (Madry et al.).
    """
    with torch.enable_grad():
        # 1. Random uniform start inside the L_inf ball [x - eps, x + eps] clamped to [0, 1]
        adv_images = images.clone().detach()
        if random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        # 2. Iterative gradient ascent steps
        for _ in range(steps):
            adv_images.requires_grad_(True)
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, targets)
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            
            # Step in sign direction of gradient
            adv_images = adv_images.detach() + alpha * grad.sign()
            
            # Projection onto L_inf ball B(images, eps) and valid image bounds [0, 1]
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()

    return adv_images