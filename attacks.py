import torch
import torch.nn.functional as F
from torch.amp import autocast

def pgd_attack(model, images, targets, eps=8/255, alpha=2/255, steps=7, random_start=True):
    with torch.enable_grad():
        adv_images = images.clone().detach()
        if random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

        for _ in range(steps):
            adv_images.requires_grad_(True)
            with autocast('cuda'):
                outputs = model(adv_images)
                loss = F.cross_entropy(outputs, targets)
            
            grad = torch.autograd.grad(loss, adv_images)[0]
            adv_images = adv_images.detach() + alpha * grad.sign()
            
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()

    return adv_images