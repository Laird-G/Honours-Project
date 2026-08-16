from attacks import pgd_attack

def clean_step(model, x, y, criterion):
    outputs = model(x)
    loss = criterion(outputs, y)
    return loss, outputs

def pgd_at_step(model, x, y, criterion, eps=8/255, alpha=2/255, steps=7):
    # Craft adversary against model in current state
    x_adv = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=True)
    outputs = model(x_adv)
    loss = criterion(outputs, y)
    return loss, outputs

# Dictionary to dispatch execution in trainer
METHODS = {
    "clean": clean_step,
    "pgd_at": pgd_at_step,
    # Add your next two training methods here:
    # "method_3": method_3_step,
    # "method_4": method_4_step,
}