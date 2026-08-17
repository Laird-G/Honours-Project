import torch
from attacks import pgd_attack

def step_clean(model, x, y, criterion, task_id=0):
    outputs = model(x, task_id=task_id)
    loss = criterion(outputs, y)
    return loss, outputs

def step_pgd(model, x, y, criterion, task_id=0, eps=8/255, alpha=2/255, steps=7):
    """PGD-AT inner maximisation.

    eps is the threat model and must stay fixed at 8/255 for numbers to be
    comparable across arms; alpha and steps are properties of the *solver* used
    during training, so they are legitimate hyperparameters (see tune.py).
    """
    x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True, task_id=task_id)
    outputs = model(x_adv, task_id=task_id)
    loss = criterion(outputs, y)
    return loss, outputs

METHODS = {
    "clean": step_clean,
    "pgd_at": step_pgd,
}
