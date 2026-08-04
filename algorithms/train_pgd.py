import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter, calculate_accuracy
from attacks.pgd import PGDAttack

def train_one_epoch_pgd(epoch, epochs, model, dataloader, criterion, optimizer, device, attack):
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

        # 1. Inner Maximization: Generate PGD-7 adversarial batch
        adv_inputs = attack.perturb(inputs, targets)

        # 2. Outer Minimization: Train model parameters on adversarial batch
        optimizer.zero_grad()
        outputs = model(adv_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        pbar.set_postfix(adv_loss=f"{losses.avg:.4f}", adv_acc=f"{accs.avg * 100:.2f}%")

    return losses.avg, accs.avg


def evaluate_robustness(epoch, epochs, model, dataloader, criterion, device, attack_20step):
    model.eval()
    clean_accs = AverageMeter()
    pgd_accs = AverageMeter()

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Eval Clean & PGD-20", 
        leave=False, 
        dynamic_ncols=True
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Evaluate Clean Accuracy
        with torch.no_grad():
            clean_outputs = model(inputs)
            clean_acc = calculate_accuracy(clean_outputs, targets)
            clean_accs.update(clean_acc, inputs.size(0))

        # Evaluate PGD-20 Robust Accuracy
        adv_inputs = attack_20step.perturb(inputs, targets)
        with torch.no_grad():
            pgd_outputs = model(adv_inputs)
            pgd_acc = calculate_accuracy(pgd_outputs, targets)
            pgd_accs.update(pgd_acc, inputs.size(0))

        pbar.set_postfix(clean_acc=f"{clean_accs.avg * 100:.2f}%", pgd20_acc=f"{pgd_accs.avg * 100:.2f}%")

    return clean_accs.avg, pgd_accs.avg


def train_pgd_model(model, trainloader, testloader, device, epochs=200):
    print(" -> Setting up CrossEntropyLoss, SGD (lr=0.1, mom=0.9, wd=5e-4)...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Attacks: 7-step for training, 20-step for robust evaluation
    train_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=7)
    eval_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=20)

    best_pgd_acc = 0.0

    print("-" * 75, flush=True)
    print(" Starting PGD-7 Adversarial Training (Madry et al. Baseline)", flush=True)
    print("-" * 75, flush=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_pgd(
            epoch, epochs, model, trainloader, criterion, optimizer, device, train_attack
        )
        clean_acc, pgd20_acc = evaluate_robustness(
            epoch, epochs, model, testloader, criterion, device, eval_attack
        )

        scheduler.step()

        saved_text = ""
        if pgd20_acc > best_pgd_acc:
            best_pgd_acc = pgd20_acc
            torch.save(model.state_dict(), "best_pgd_at_model.pth")
            saved_text = " -> Saved Best Robust Model!"

        print(
            f"Epoch [{epoch+1:03d}/{epochs:03d}] "
            f"| Train Adv Loss: {train_loss:.4f} - Train Adv Acc: {train_acc*100:.2f}% "
            f"| Test Clean Acc: {clean_acc*100:.2f}% - Test PGD-20 Acc: {pgd20_acc*100:.2f}%"
            f"{saved_text}",
            flush=True
        )

    print("-" * 75, flush=True)
    print(f"Training complete. Best PGD-20 Robust Accuracy: {best_pgd_acc*100:.2f}%", flush=True)