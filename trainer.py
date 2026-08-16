import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack

def evaluate(model, loader, device, attack_steps=0, eps=8/255, alpha=2/255, desc="Eval"):
    model.eval()
    correct, total = 0, 0
    
    # Added tqdm to display progress during evaluation
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        if attack_steps > 0:
            x = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=attack_steps, random_start=True)
        with torch.no_grad():
            outputs = model(x)
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")

    return correct / total

def train(model, trainloader, testloader, optimizer, scheduler, criterion, 
          step_fn, epochs, device, save_name="best_model.pth"):
    scaler = GradScaler('cuda')
    best_robust_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(trainloader, desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Train", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                loss, outputs = step_fn(model, x, y, criterion)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(loss=f"{train_loss/train_total:.3f}", acc=f"{100*train_correct/train_total:.1f}%")

        scheduler.step()
        clean_acc = evaluate(model, testloader, device, attack_steps=0, desc="Eval Clean")
        
        pgd20_str, save_flag = "", ""
        # Run PGD-20 evaluation every 10 epochs (or the final epoch of a full run)
        if (epoch + 1) % 10 == 0 or ((epoch + 1) == epochs and epochs > 1):
            pgd20_acc = evaluate(model, testloader, device, attack_steps=20, desc="Eval PGD-20")
            pgd20_str = f" | PGD-20: {100*pgd20_acc:.2f}%"
            if pgd20_acc > best_robust_acc:
                best_robust_acc = pgd20_acc
                torch.save(model.state_dict(), save_name)
                save_flag = " -> Saved Checkpoint!"

        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss/train_total:.3f} "
              f"| Clean Acc: {100*clean_acc:.2f}%{pgd20_str}{save_flag}", flush=True)