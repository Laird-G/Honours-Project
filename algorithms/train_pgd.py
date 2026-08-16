import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter, calculate_accuracy
from attacks.pgd import PGDAttack
from torch.amp import autocast, GradScaler

def train_one_epoch_pgd(epoch, epochs, model, dataloader, criterion, optimizer, device, attack, scaler):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] PGD-7 Train", 
        leave=False, 
        dynamic_ncols=True
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        adv_inputs = attack.perturb(inputs, targets)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        pbar.set_postfix(adv_loss=f"{losses.avg:.4f}", adv_acc=f"{accs.avg * 100:.2f}%")

    return losses.avg, accs.avg


def evaluate_clean(epoch, epochs, model, dataloader, criterion, device):
    model.eval()
    clean_accs = AverageMeter()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast('cuda'):
                outputs = model(inputs)
            clean_acc = calculate_accuracy(outputs, targets)
            clean_accs.update(clean_acc, inputs.size(0))

    return clean_accs.avg


def evaluate_robustness(epoch, epochs, model, dataloader, criterion, device, attack_20step):
    model.eval()
    pgd_accs = AverageMeter()

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Eval PGD-20", 
        leave=False, 
        dynamic_ncols=True
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = attack_20step.perturb(inputs, targets)
        
        with torch.no_grad():
            with autocast('cuda'):
                pgd_outputs = model(adv_inputs)
            pgd_acc = calculate_accuracy(pgd_outputs, targets)
            pgd_accs.update(pgd_acc, inputs.size(0))

        pbar.set_postfix(pgd20_acc=f"{pgd_accs.avg * 100:.2f}%")

    return pgd_accs.avg


def train_pgd_model(model, trainloader, testloader, device, epochs=200):
    print(" -> Setting up CrossEntropyLoss, SGD (lr=0.1, mom=0.9, wd=5e-4), AMP Scaler...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    train_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=7)
    eval_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=20)

    best_pgd_acc = 0.0

    print("-" * 75, flush=True)
    print(" Starting Accelerated PGD-7 Adversarial Training (AMP Enabled)", flush=True)
    print("-" * 75, flush=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_pgd(
            epoch, epochs, model, trainloader, criterion, optimizer, device, train_attack, scaler
        )
        
        clean_acc = evaluate_clean(epoch, epochs, model, testloader, criterion, device)
        scheduler.step()

        pgd20_str = "PGD-20: Skipped (Runs every 10 epochs)"
        saved_text = ""

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            pgd20_acc = evaluate_robustness(epoch, epochs, model, testloader, criterion, device, eval_attack)
            pgd20_str = f"Test PGD-20 Acc: {pgd20_acc*100:.2f}%"

            if pgd20_acc > best_pgd_acc:
                best_pgd_acc = pgd20_acc
                torch.save(model.state_dict(), "best_pgd_at_model.pth")
                saved_text = " -> Saved Best Robust Model!"

        print(
            f"Epoch [{epoch+1:03d}/{epochs:03d}] "
            f"| Train Adv Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
            f"| Test Clean Acc: {clean_acc*100:.2f}% | {pgd20_str}"
            f"{saved_text}",
            flush=True
        )

    print("-" * 75, flush=True)
    print(f"Training complete. Best PGD-20 Robust Accuracy: {best_pgd_acc*100:.2f}%", flush=True)