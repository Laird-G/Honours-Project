import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack
from gpm import get_gpm_bases, project_backbone_gradients

def evaluate(model, loader, device, task_id=0, attack_steps=0, eps=8/255, alpha=2/255, desc="Eval"):
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if attack_steps > 0:
            x = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=attack_steps, random_start=True, task_id=task_id)
        with torch.no_grad():
            with autocast('cuda'):
                outputs = model(x, task_id=task_id)
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        pbar.set_postfix(acc=f"{100 * correct / total:.2f}%")
    return correct / total

def train_standard(model, trainloader, testloader, optimizer, scheduler, criterion,
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
                loss, outputs = step_fn(model, x, y, criterion, task_id=0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(loss=f"{train_loss/train_total:.3f}", acc=f"{100*train_correct/train_total:.1f}%")

        scheduler.step()
        clean_acc = evaluate(model, testloader, device, task_id=0, attack_steps=0, desc="Eval Clean")

        eval_str, save_flag = "", ""
        if not is_adversarial:
            if clean_acc > best_metric:
                best_metric = clean_acc
                torch.save(model.state_dict(), save_name)
                save_flag = " -> Saved Best Clean Model!"
        else:
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                pgd20_acc = evaluate(model, testloader, device, task_id=0, attack_steps=20, desc="Eval PGD-20")
                eval_str = f" | PGD-20: {100*pgd20_acc:.2f}%"
                if pgd20_acc > best_metric:
                    best_metric = pgd20_acc
                    torch.save(model.state_dict(), save_name)
                    save_flag = " -> Saved Best Robust Model!"

        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss/train_total:.3f} "
              f"| Clean Acc: {100*clean_acc:.2f}%{eval_str}{save_flag}", flush=True)

def train_gpm_pipeline(model, trainloader, valloader, testloader, criterion,
                       threshold, clean_checkpoint, epochs_task1, epochs_task2, device, save_name):
    scaler = GradScaler('cuda')

    # ---------------------------------------------------------
    # STAGE 1: Clean Baseline Training (or Checkpoint Loading)
    # ---------------------------------------------------------
    if clean_checkpoint and os.path.exists(clean_checkpoint):
        print(f"\n[STAGE 1/3] Loading pre-trained clean checkpoint: {clean_checkpoint}", flush=True)
        state_dict = torch.load(clean_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 1/3] Training Clean Task 1 ({epochs_task1} Epochs)", flush=True)
        print("=" * 70, flush=True)

        model.base_model.heads[1].requires_grad_(False)
        opt_t1 = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched_t1 = optim.lr_scheduler.MultiStepLR(opt_t1, milestones=[100, 125], gamma=0.1)

        for epoch in range(epochs_task1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            pbar = tqdm(trainloader, desc=f"Task 1 [{epoch+1:03d}/{epochs_task1:03d}]", leave=False, dynamic_ncols=True)
            for x, y in pbar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt_t1.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    outputs = model(x, task_id=0)
                    loss = criterion(outputs, y)

                scaler.scale(loss).backward()
                scaler.step(opt_t1)
                scaler.update()

                train_loss += loss.item() * y.size(0)
                train_correct += (outputs.argmax(dim=1) == y).sum().item()
                train_total += y.size(0)
                pbar.set_postfix(loss=f"{train_loss/train_total:.3f}", acc=f"{100*train_correct/train_total:.1f}%")

            sched_t1.step()
            if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs_task1:
                acc = evaluate(model, testloader, device, task_id=0, attack_steps=0, desc="Eval Clean")
                print(f"Task 1 Epoch [{epoch+1:03d}/{epochs_task1:03d}] | Clean Acc: {acc*100:.2f}%", flush=True)

    # ---------------------------------------------------------
    # STAGE 2: SVD Representation Extraction
    # ---------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 2/3] Extracting SVD Bases (Threshold l_th = {threshold})", flush=True)
    print("=" * 70, flush=True)
    gpm_bases = get_gpm_bases(model, valloader, device, threshold=threshold, task_id=0)
    print(f" -> Extracted orthogonal basis tensors for {len(gpm_bases)} convolution layers.", flush=True)

    # ---------------------------------------------------------
    # STAGE 3: GPM-Constrained Adversarial Training (Task 2)
    # ---------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 3/3] GPM-Constrained PGD-10 Adversarial Training (Max {epochs_task2} Epochs)", flush=True)
    print("=" * 70, flush=True)

    model.base_model.heads[0].requires_grad_(False)
    model.base_model.heads[1].requires_grad_(True)

    opt_t2 = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.0, weight_decay=0.0)
    sched_t2 = optim.lr_scheduler.ReduceLROnPlateau(opt_t2, mode='max', factor=1/3, patience=5, min_lr=1e-7)

    best_val_acc = 0.0
    for epoch in range(epochs_task2):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(trainloader, desc=f"Task 2 [{epoch+1:03d}/{epochs_task2:03d}]", leave=False, dynamic_ncols=True)

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_adv = pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=10, random_start=True, task_id=1)

            opt_t2.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(x_adv, task_id=1)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt_t2)

            project_backbone_gradients(model, gpm_bases)

            scaler.step(opt_t2)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(adv_loss=f"{train_loss/train_total:.3f}", adv_acc=f"{100*train_correct/train_total:.1f}%")

        val_acc = evaluate(model, valloader, device, task_id=1, attack_steps=10, desc="Val PGD-10")
        sched_t2.step(val_acc)
        current_lr = opt_t2.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_name)

        if (epoch + 1) % 5 == 0 or current_lr < 1e-7 or (epoch + 1) == epochs_task2:
            print(f"Task 2 Epoch [{epoch+1:03d}/{epochs_task2:03d}] | Val Adv Acc: {val_acc*100:.2f}% | LR: {current_lr:.2e}", flush=True)

        if current_lr < 1e-7:
            print(f" -> Learning rate floor reached. Converged at epoch {epoch+1}.", flush=True)
            break

    # Final Benchmark Evaluation
    print("\n" + "=" * 70, flush=True)
    print(" Final Benchmark Verification", flush=True)
    print("=" * 70, flush=True)
    final_clean = evaluate(model, testloader, device, task_id=0, attack_steps=0, desc="Final Clean Head 0")
    final_pgd20 = evaluate(model, testloader, device, task_id=1, attack_steps=20, desc="Final PGD-20 Head 1")
    print(f"Threshold (l_th)      : {threshold}", flush=True)
    print(f"Preserved Clean Acc   : {final_clean * 100:.2f}%", flush=True)
    print(f"PGD-20 Robust Acc     : {final_pgd20 * 100:.2f}%", flush=True)