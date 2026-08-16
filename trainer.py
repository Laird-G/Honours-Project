import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack

def evaluate(model, loader, device, attack_steps=0, eps=8/255, alpha=2/255, desc="Eval"):
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if attack_steps > 0:
            x = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=attack_steps, random_start=True)
        with torch.no_grad():
            with autocast('cuda'):
                outputs = model(x)
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")
    return correct / total

def train(model, trainloader, testloader, optimizer, scheduler, criterion, 
          step_fn, mode, epochs, device, save_name="best_model.pth"):
    scaler = GradScaler('cuda')
    best_metric = 0.0
    is_adversarial = (mode != "clean")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(trainloader, desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Train", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

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
        
        eval_str, save_flag = "", ""
        
        if not is_adversarial:
            # Clean mode: only evaluate clean test accuracy and checkpoint when improved
            if clean_acc > best_metric:
                best_metric = clean_acc
                torch.save(model.state_dict(), save_name)
                save_flag = " -> Saved Best Clean Model!"
        else:
            # Adversarial mode: evaluate PGD-20 every 10 epochs and on the final epoch
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                pgd20_acc = evaluate(model, testloader, device, attack_steps=20, desc="Eval PGD-20")
                eval_str = f" | PGD-20: {100*pgd20_acc:.2f}%"
                if pgd20_acc > best_metric:
                    best_metric = pgd20_acc
                    torch.save(model.state_dict(), save_name)
                    save_flag = " -> Saved Best Robust Model!"

        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss/train_total:.3f} "
              f"| Clean Acc: {100*clean_acc:.2f}%{eval_str}{save_flag}", flush=True)