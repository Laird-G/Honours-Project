import torch
import torch.nn as nn

class PGDAttack:
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=7):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images, targets):
        """
        Generates l_infinity PGD adversarial examples.
        """
        is_training = self.model.training
        self.model.eval()

        # 1. Random uniform initialization within [-epsilon, +epsilon]
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()

        # 2. Iterative gradient ascent
        for _ in range(self.steps):
            adv_images.requires_grad_()
            
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, targets)

            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            # Projection step
            eta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()

        if is_training:
            self.model.train()

        return adv_images