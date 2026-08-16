import torch
from attacks import pgd_attack

def step_clean(model, x, y, criterion, task_id=0):
    outputs = model(x, task_id=task_id)
    loss = criterion(outputs, y)
    return loss, outputs

def step_pgd(model, x, y, criterion, task_id=0):
    x_adv = pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=7, random_start=True, task_id=task_id)
    outputs = model(x_adv, task_id=task_id)
    loss = criterion(outputs, y)
    return loss, outputs

METHODS = {
    "clean": step_clean,
    "pgd_at": step_pgd,
}