import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack
from gpm import get_gpm_bases, project_backbone_gradients
from ogp import (gram_schmidt, project_orthogonal, reference_gradients,
                 selfcheck as ogp_selfcheck, flat_dot, flat_norm)

MIN_LR = 1e-7
PROJ_NORM_LOG_EVERY = 50


def load_clean_checkpoint(model, path, device):
    """Load a Task 1 / pre-alignment checkpoint, refusing partial matches[cite: 3]."""
    state_dict = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch -- refusing to run on a partially "
            f"loaded backbone.\n  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
        )
    print(f" -> Loaded {len(state_dict)} tensors from {path}, all keys matched.", flush=True)


def freeze_backbone_bn(model):
    """Freeze every BatchNorm in the shared backbone f[cite: 3].

    Stops two failure modes during adversarial training:
      1. Affine weight/bias receiving unprojected gradients[cite: 3].
      2. Running_mean/running_var updating on adversarial batches inside forward()[cite: 3].
    """
    for module in model.base_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)


def evaluate(model, loader, device, task_id=0, attack_steps=0, eps=8/255, alpha=2/255, desc="Eval",
             max_batches=None):
    """Standard multi-step evaluation helper[cite: 3]."""
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for i, (x, y) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
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
                   step_fn, mode, epochs, device, save_name="best_model.pth",
                   step_kwargs=None, eval_every=10, max_eval_batches=None,
                   save_best=True, verbose=True):
    """Standard single-task loop[cite: 3]."""
    scaler = GradScaler('cuda')
    best_metric = 0.0
    is_adversarial = (mode != "clean")
    step_kwargs = step_kwargs or {}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(trainloader, desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Train", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                loss, outputs = step_fn(model, x, y, criterion, task_id=0, **step_kwargs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(loss=f"{train_loss/train_total:.3f}", acc=f"{100*train_correct/train_total:.1f}%")

        scheduler.step()
        clean_acc = evaluate(model, testloader, device, task_id=0, attack_steps=0, desc="Eval Clean",
                             max_batches=max_eval_batches)

        eval_str, save_flag = "", ""
        if not is_adversarial:
            if clean_acc > best_metric:
                best_metric = clean_acc
                if save_best:
                    torch.save(model.state_dict(), save_name)
                    save_flag = " -> Saved Best Clean Model!"
        else:
            if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
                pgd20_acc = evaluate(model, testloader, device, task_id=0, attack_steps=20, desc="Eval PGD-20",
                                     max_batches=max_eval_batches)
                eval_str = f" | PGD-20: {100*pgd20_acc:.2f}%"
                if pgd20_acc > best_metric:
                    best_metric = pgd20_acc
                    if save_best:
                        torch.save(model.state_dict(), save_name)
                        save_flag = " -> Saved Best Robust Model!"

        if verbose:
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Loss: {train_loss/train_total:.3f} "
                  f"| Clean Acc: {100*clean_acc:.2f}%{eval_str}{save_flag}", flush=True)

    if is_adversarial and verbose:
        print(f"\nAdversarial epochs completed: {epochs} | Best PGD-20: {100*best_metric:.2f}%", flush=True)

    return best_metric


def train_gpm_pipeline(model, trainloader, valloader, testloader, criterion,
                       threshold, clean_checkpoint, epochs_task1, epochs_task2, device, save_name,
                       gpm_samples=None, no_oracle_eval=False,
                       lr_task1=0.1, lr_task2=0.01, plateau_factor=1/3, plateau_patience=5,
                       adv_alpha=2/255, adv_steps=10, save_best=True, verbose=True,
                       final_eval_loader=None, final_attack_steps=20, max_eval_batches=None):
    """Three-stage GPM pipeline[cite: 3]."""
    scaler = GradScaler('cuda')
    final_eval_loader = final_eval_loader if final_eval_loader is not None else testloader

    if clean_checkpoint and os.path.exists(clean_checkpoint):
        print(f"\n[STAGE 1/3] Loading pre-trained clean checkpoint: {clean_checkpoint}", flush=True)
        load_clean_checkpoint(model, clean_checkpoint, device)
    else:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 1/3] Training Clean Task 1 ({epochs_task1} Epochs)", flush=True)
        print("=" * 70, flush=True)

        model.base_model.heads[1].requires_grad_(False)
        opt_t1 = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr_task1, momentum=0.9, weight_decay=5e-4)
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

        torch.save(model.state_dict(), "task1_clean_backbone.pth")
        print(" -> Saved Task 1 backbone to task1_clean_backbone.pth", flush=True)

    task1_clean_acc = evaluate(model, final_eval_loader, device, task_id=0, attack_steps=0,
                               desc="Task 1 Clean", max_batches=max_eval_batches)
    print(f" -> Task 1 clean accuracy (pre-GPM reference): {task1_clean_acc*100:.2f}%", flush=True)

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 2/3] Extracting SVD Bases (Threshold l_th = {threshold})", flush=True)
        print("=" * 70, flush=True)
    gpm_bases, gpm_stats = get_gpm_bases(
        model, valloader, device, threshold=threshold, task_id=0,
        max_samples=gpm_samples, return_stats=True,
    )
    total_k = sum(k for k, _ in gpm_stats.values())
    total_n = sum(n for _, n in gpm_stats.values())
    if verbose:
        print(f" -> Extracted orthogonal basis tensors for {len(gpm_bases)} convolution layers.", flush=True)
        print(f" -> Salient subspace size (k/n) per layer at l_th = {threshold}:", flush=True)
        for name, (k, n) in gpm_stats.items():
            print(f"      {name:<40s} k={k:>5d} / n={n:>5d}  ({100.0*k/n:5.1f}%)", flush=True)
    print(f" -> Aggregate: k/n = {total_k}/{total_n} ({100.0*total_k/total_n:.1f}%)", flush=True)

    if verbose:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 3/3] GPM-Constrained PGD-{adv_steps} Adversarial Training "
              f"(Max {epochs_task2} Epochs)", flush=True)
        print("=" * 70, flush=True)

    model.base_model.heads[0].requires_grad_(False)
    model.base_model.heads[1].requires_grad_(True)
    freeze_backbone_bn(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_t2 = optim.SGD(trainable, lr=lr_task2, momentum=0.0, weight_decay=0.0)
    sched_t2 = optim.lr_scheduler.ReduceLROnPlateau(opt_t2, mode='max', factor=plateau_factor,
                                                    patience=plateau_patience, min_lr=MIN_LR)

    best_val_acc = -1.0
    epochs_run = 0
    for epoch in range(epochs_task2):
        model.train()
        freeze_backbone_bn(model)
        epochs_run = epoch + 1

        train_loss, train_correct, train_total = 0.0, 0, 0
        norm_ratio_sum, norm_ratio_n = 0.0, 0
        pbar = tqdm(trainloader, desc=f"Task 2 [{epoch+1:03d}/{epochs_task2:03d}]", leave=False, dynamic_ncols=True)

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_adv = pgd_attack(model, x, y, eps=8/255, alpha=adv_alpha, steps=adv_steps,
                               random_start=True, task_id=1)

            opt_t2.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(x_adv, task_id=1)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt_t2)

            log_norms = (step % PROJ_NORM_LOG_EVERY == 0)
            norms = project_backbone_gradients(model, gpm_bases, return_norms=log_norms)
            if norms:
                ratios = [after / before for before, after in norms.values() if before > 0]
                if ratios:
                    norm_ratio_sum += sum(ratios) / len(ratios)
                    norm_ratio_n += 1

            scaler.step(opt_t2)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(adv_loss=f"{train_loss/train_total:.3f}", adv_acc=f"{100*train_correct/train_total:.1f}%")

        val_acc = evaluate(model, valloader, device, task_id=1, attack_steps=10, desc="Val PGD-10",
                           max_batches=max_eval_batches)
        sched_t2.step(val_acc)
        current_lr = opt_t2.param_groups[0]['lr']
        proj_ratio = (norm_ratio_sum / norm_ratio_n) if norm_ratio_n else float('nan')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_best:
                torch.save(model.state_dict(), save_name)

        converged = current_lr <= MIN_LR * (1 + 1e-6)
        if verbose and ((epoch + 1) % 5 == 0 or converged or (epoch + 1) == epochs_task2):
            print(f"Task 2 Epoch [{epoch+1:03d}/{epochs_task2:03d}] | Val Adv Acc: {val_acc*100:.2f}% "
                  f"| LR: {current_lr:.2e} | ||G_ortho||/||G||: {proj_ratio:.4f}", flush=True)

        if converged:
            print(f" -> Learning rate floor ({MIN_LR:.0e}) reached. Converged at epoch {epoch+1}.", flush=True)
            break

    if save_best and os.path.exists(save_name):
        print(f" -> Restoring best-validation checkpoint: {save_name}", flush=True)
        model.load_state_dict(torch.load(save_name, map_location=device))

    final_clean = evaluate(model, final_eval_loader, device, task_id=0, attack_steps=0,
                           desc="Final Clean Head 0", max_batches=max_eval_batches)
    final_pgd20 = evaluate(model, final_eval_loader, device, task_id=1, attack_steps=final_attack_steps,
                           desc=f"Final PGD-{final_attack_steps} Head 1", max_batches=max_eval_batches)

    if no_oracle_eval:
        h1_clean = evaluate(model, final_eval_loader, device, task_id=1, attack_steps=0, desc="Clean Head 1")
        h0_pgd20 = evaluate(model, final_eval_loader, device, task_id=0, attack_steps=20, desc="PGD-20 Head 0")
        print("\n" + "-" * 70, flush=True)
        print(" Deployable read without task oracle:", flush=True)
        print(f"   Head 1 Clean: {h1_clean*100:.2f}% | Head 1 PGD-20: {final_pgd20*100:.2f}%", flush=True)
        print("-" * 70, flush=True)

    return {
        "threshold": threshold,
        "k": total_k,
        "n": total_n,
        "task1_clean": task1_clean_acc,
        "final_clean": final_clean,
        "final_robust": final_pgd20,
        "best_val_robust": best_val_acc,
        "epochs_run": epochs_run,
    }


def train_ogp_pipeline(model, trainloader, valloader, testloader, criterion, ref_loaders,
                       clean_checkpoint, epochs, lr, refresh_every, delta, warmup_ratio,
                       adv_alpha, adv_steps, device, save_name, task_id=0,
                       anchor_weight=0.01, ref_temp=2.0, verbose=True):
    """OGPSA with Vision Stability Fixes[cite: 1, 3].

    KEY IMPROVEMENTS:
    1. FREEZE BATCHNORM: running_mean/var are frozen so adversarial perturbations
       cannot overwrite clean normalization statistics[cite: 3].
    2. L2 WEIGHT ANCHOR: Adds (anchor_weight)*||theta - theta_pre|| to bound higher-order
       Hessian curvature drift outside the clean basin[cite: 1].
    3. TEMPERATURE-SCALED REFERENCE GRADIENTS: Prevents zero-gradient collapse at clean convergence[cite: 3].
    """
    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 1/2] Loading pre-alignment checkpoint: {clean_checkpoint}", flush=True)
    print("=" * 70, flush=True)
    load_clean_checkpoint(model, clean_checkpoint, device)

    # Use single Head 0 for a directly deployable classifier
    model.base_model.heads[1].requires_grad_(False)
    model.base_model.heads[0].requires_grad_(True)

    # CRITICAL FIX 1: Freeze BatchNorm running stats and affine parameters[cite: 3]
    freeze_backbone_bn(model)

    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f" -> BatchNorm frozen. Projecting over {len(params)} tensors / {n_param:,} parameters.", flush=True)

    # Cache pre-trained weights for the L2 proximity anchor[cite: 1]
    theta_pre = [p.detach().clone() for p in params]

    pre_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0, desc="Pre-align Clean")
    print(f" -> Pre-alignment clean accuracy: {pre_clean*100:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 2/2] OGP-Constrained PGD-{adv_steps} Fine-Tune ({epochs} Epochs, "
          f"K = {refresh_every}, M = {len(ref_loaders)}, Anchor = {anchor_weight}, Temp = {ref_temp})", flush=True)
    print("=" * 70, flush=True)

    opt = optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)

    steps_per_epoch = len(trainloader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler('cuda')

    basis, residuals = [], []
    refreshes, skipped_refreshes = 0, 0
    diagnosed, checked = False, False
    global_step = 0

    for epoch in range(epochs):
        model.train()
        # Must re-freeze BN because model.train() enables all submodules[cite: 3]
        freeze_backbone_bn(model)

        train_loss, train_correct, train_total = 0.0, 0, 0
        ratio_sum, ratio_n = 0.0, 0
        last_cosines = []

        pbar = tqdm(trainloader, desc=f"OGP [{epoch+1:03d}/{epochs:03d}]", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            # Refresh reference subspace with temperature-scaled gradients[cite: 1, 2, 3]
            if global_step % refresh_every == 0:
                grads = reference_gradients(
                    model, ref_loaders, criterion, params, device,
                    task_id=task_id, temperature=ref_temp
                )
                ref_norms = [flat_norm(g) for g in grads]

                if all(math.isfinite(n) for n in ref_norms):
                    if not diagnosed:
                        print(f" -> First refresh: ||g_ref|| = "
                              f"{', '.join(f'{n:.3e}' for n in ref_norms)}", flush=True)
                        diagnosed = True

                    basis, residuals = gram_schmidt(grads, delta=delta)
                    refreshes += 1
                else:
                    skipped_refreshes += 1
                del grads

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_adv = pgd_attack(model, x, y, eps=8/255, alpha=adv_alpha, steps=adv_steps,
                               random_start=True, task_id=task_id)

            opt.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = model(x_adv, task_id=task_id)
                loss = criterion(outputs, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            # Orthogonal projection[cite: 1, 2]
            if basis:
                want_stats = (global_step % PROJ_NORM_LOG_EVERY == 0) or not checked
                stats = project_orthogonal(basis, params, return_stats=want_stats)
                if stats is not None:
                    before, after, cosines = stats
                    if before > 0:
                        ratio_sum += after / before
                        ratio_n += 1
                        last_cosines = cosines

                if not checked:
                    gram_err, resid, ok = ogp_selfcheck(basis, params)
                    print(f" -> Self-check: max|U^T U - I| = {gram_err:.2e}, "
                          f"max|<u_j, g'>|/||g'|| = {resid:.2e} -> "
                          f"{'PASS' if ok else 'WARN'}", flush=True)
                    checked = True

            # CRITICAL FIX 2: L2 Proximity Anchor (Curvature Barrier)[cite: 1]
            if anchor_weight > 0:
                with torch.no_grad():
                    for p, p0 in zip(params, theta_pre):
                        if p.grad is not None:
                            p.grad.add_(p - p0, alpha=anchor_weight)

            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(adv_loss=f"{train_loss/train_total:.3f}",
                             adv_acc=f"{100*train_correct/train_total:.1f}%")

        val_clean = evaluate(model, valloader, device, task_id=task_id, attack_steps=0, desc="Val Clean")
        val_adv = evaluate(model, valloader, device, task_id=task_id, attack_steps=10, desc="Val PGD-10")

        # Track Euclidean parameter distance from the clean checkpoint
        with torch.no_grad():
            dist_from_pre = math.sqrt(sum((p - p0).norm().item()**2 for p, p0 in zip(params, theta_pre)))

        if verbose:
            proj_ratio = (ratio_sum / ratio_n) if ratio_n else float('nan')
            cos_str = ", ".join(f"{c:+.3f}" for c in last_cosines[:4]) if last_cosines else "n/a"
            print(f"OGP Epoch [{epoch+1:03d}/{epochs:03d}] | Val Clean: {val_clean*100:.2f}% "
                  f"| Val PGD-10: {val_adv*100:.2f}% | ||theta-theta_0||: {dist_from_pre:.3f} "
                  f"| LR: {opt.param_groups[0]['lr']:.2e} | ||g'||/||g||: {proj_ratio:.4f} "
                  f"| rank M'={len(basis)} | cos(g, u): [{cos_str}]", flush=True)

    torch.save(model.state_dict(), save_name)
    print(f" -> Saved final model to {save_name}", flush=True)

    final_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0, desc="Final Clean")
    final_pgd20 = evaluate(model, testloader, device, task_id=task_id, attack_steps=20, desc="Final PGD-20")

    print("\n" + "=" * 70, flush=True)
    print(" Final Benchmark Verification (Single Deployable Head)", flush=True)
    print("=" * 70, flush=True)
    print(f"Refresh period K      : {refresh_every}", flush=True)
    print(f"Reference sets M      : {len(ref_loaders)} (rank M' = {len(basis)})", flush=True)
    print(f"Pre-align Clean Acc   : {pre_clean*100:.2f}%", flush=True)
    print(f"Preserved Clean Acc   : {final_clean*100:.2f}% (delta: {100*(final_clean-pre_clean):+.2f}%)", flush=True)
    print(f"PGD-20 Robust Acc     : {final_pgd20*100:.2f}%", flush=True)

    return {
        "refresh_every": refresh_every,
        "num_refs": len(ref_loaders),
        "rank": len(basis),
        "pre_clean": pre_clean,
        "final_clean": final_clean,
        "final_robust": final_pgd20,
        "epochs_run": epochs,
    }