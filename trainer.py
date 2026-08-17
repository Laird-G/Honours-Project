import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack
from gpm import get_gpm_bases, project_backbone_gradients

# Sec. 4.4.2: "training process continues until learning rate reaches a
# minimum value of 1e-7". ReduceLROnPlateau clamps *at* min_lr, so the
# stopping test has to be <=, not <.
MIN_LR = 1e-7
# How often to record ||G|| before/after projection (Figs. 4.5 / 4.7).
PROJ_NORM_LOG_EVERY = 50


def freeze_backbone_bn(model):
    """Freeze every BatchNorm in the shared backbone f.

    Sec. 4.3.2 imposes the GPM constraint on the whole backbone, but BatchNorm
    has no patch-space representation to project onto, so there is no basis to
    null its gradient against. Two separate leaks have to be closed:

      * affine weight/bias would otherwise receive *unprojected* adversarial
        gradient every step -- an escape hatch letting task 2 learn around the
        constraint, invariant to l_th;
      * running_mean/running_var are buffers, updated inside forward() whenever
        the module is in train mode, ignoring requires_grad, the optimizer and
        the projection entirely.

    Without this, at l_th -> 0.99 the conv gradients are nulled but BN still
    drifts, so head 0's clean accuracy collapses regardless of threshold and
    the control condition of Figs. 4.3 / 4.4 cannot be reproduced.

    Must be re-applied after every model.train(), which re-enables train mode
    on all submodules. Idempotent.
    """
    for module in model.base_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()                        # stops running-stat updates
            module.weight.requires_grad_(False)
            module.bias.requires_grad_(False)


def evaluate(model, loader, device, task_id=0, attack_steps=0, eps=8/255, alpha=2/255, desc="Eval",
             max_batches=None):
    """max_batches truncates the evaluation to the first N batches.

    Only for the hyperparameter search (tune.py), where a PGD-20 pass over the
    full split dominates the cost of a short proxy run. Reported numbers must be
    produced with max_batches=None.
    """
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
    """Standard single-task loop.

    Returns the best metric of the run (clean accuracy for mode="clean",
    PGD-20 accuracy otherwise) so a hyperparameter search can score it.

    step_kwargs is forwarded to step_fn (e.g. the training attack's alpha/steps).
    eval_every / max_eval_batches / save_best / verbose exist so tune.py can run
    a cheap proxy without writing checkpoints; defaults reproduce the full run.
    """
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
        # Logged so the adversarial-epoch budget can be compared against the
        # GPM arm, which stops on its own plateau criterion.
        print(f"\nAdversarial epochs completed: {epochs} | Best PGD-20: {100*best_metric:.2f}%", flush=True)

    return best_metric


def train_gpm_pipeline(model, trainloader, valloader, testloader, criterion,
                       threshold, clean_checkpoint, epochs_task1, epochs_task2, device, save_name,
                       gpm_samples=None, no_oracle_eval=False,
                       lr_task1=0.1, lr_task2=0.01, plateau_factor=1/3, plateau_patience=5,
                       adv_alpha=2/255, adv_steps=10, save_best=True, verbose=True,
                       final_eval_loader=None, final_attack_steps=20, max_eval_batches=None):
    """Three-stage GPM pipeline. Returns a dict of the run's headline metrics.

    The searchable knobs are threshold, lr_task2, the ReduceLROnPlateau shape
    (plateau_factor / plateau_patience), the Task 2 attack solver
    (adv_alpha / adv_steps) and gpm_samples. momentum and weight decay are
    deliberately *not* exposed: Sec. 4.4.2 fixes both at 0 because SGD applies
    them after the projection, which would re-inject the interference the
    projection removes.

    save_best / verbose / final_eval_loader / final_attack_steps /
    max_eval_batches exist for tune.py; the defaults reproduce the full run.
    """
    scaler = GradScaler('cuda')
    final_eval_loader = final_eval_loader if final_eval_loader is not None else testloader

    # ---------------------------------------------------------
    # STAGE 1: Clean Baseline Training (or Checkpoint Loading)
    # ---------------------------------------------------------
    if clean_checkpoint and os.path.exists(clean_checkpoint):
        print(f"\n[STAGE 1/3] Loading pre-trained clean checkpoint: {clean_checkpoint}", flush=True)
        state_dict = torch.load(clean_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # strict=False would otherwise let a key mismatch through silently
            # and run the entire threshold sweep on a randomly initialised
            # backbone, producing plausible-looking but meaningless curves.
            raise RuntimeError(
                f"Checkpoint key mismatch -- refusing to run the sweep on a partially "
                f"loaded backbone.\n  missing ({len(missing)}): {missing[:8]}\n"
                f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
            )
        print(f" -> Loaded {len(state_dict)} tensors, all keys matched.", flush=True)
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

    # ---------------------------------------------------------
    # STAGE 2: SVD Representation Extraction
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 2/3] Extracting SVD Bases (Threshold l_th = {threshold})", flush=True)
        print("=" * 70, flush=True)
    gpm_bases, gpm_stats = get_gpm_bases(
        model, valloader, device, threshold=threshold, task_id=0,
        max_samples=gpm_samples, return_stats=True,
    )
    # k/n per layer -- this is the data behind Fig. 4.8 and the principal
    # component regions of Fig. 4.9.
    total_k = sum(k for k, _ in gpm_stats.values())
    total_n = sum(n for _, n in gpm_stats.values())
    if verbose:
        print(f" -> Extracted orthogonal basis tensors for {len(gpm_bases)} convolution layers.", flush=True)
        print(f" -> Salient subspace size (k/n) per layer at l_th = {threshold}:", flush=True)
        for name, (k, n) in gpm_stats.items():
            print(f"      {name:<40s} k={k:>5d} / n={n:>5d}  ({100.0*k/n:5.1f}%)", flush=True)
    print(f" -> Aggregate: k/n = {total_k}/{total_n} ({100.0*total_k/total_n:.1f}%)", flush=True)

    # ---------------------------------------------------------
    # STAGE 3: GPM-Constrained Adversarial Training (Task 2)
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70, flush=True)
        print(f"[STAGE 3/3] GPM-Constrained PGD-{adv_steps} Adversarial Training "
              f"(Max {epochs_task2} Epochs)", flush=True)
        print("=" * 70, flush=True)

    model.base_model.heads[0].requires_grad_(False)
    model.base_model.heads[1].requires_grad_(True)
    # Before building the optimizer, so BN affine params are excluded from it.
    freeze_backbone_bn(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f" -> Backbone BatchNorm frozen (affine + running stats). "
              f"{len(trainable)} trainable tensors in Task 2.", flush=True)

    # Sec. 4.4.2: momentum = 0 and weight decay = 0, "in order to avoid
    # distorting the gradient direction after projecting in the orthogonal
    # direction of the salient space". Weight decay in particular is applied
    # inside SGD.step(), i.e. *after* the projection, and w has a large
    # component inside the salient space -- it would re-inject exactly the
    # interference the projection removes.
    opt_t2 = optim.SGD(trainable, lr=lr_task2, momentum=0.0, weight_decay=0.0)
    sched_t2 = optim.lr_scheduler.ReduceLROnPlateau(opt_t2, mode='max', factor=plateau_factor,
                                                    patience=plateau_patience, min_lr=MIN_LR)

    best_val_acc = -1.0        # so epoch 1 always writes a checkpoint
    epochs_run = 0
    for epoch in range(epochs_task2):
        model.train()
        freeze_backbone_bn(model)      # model.train() just re-enabled them
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
            # Unscale before projecting so the basis and the gradient are on the
            # same scale, and so scaler.step() does not unscale a second time.
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
            # proj_ratio is mean_layers(||G'|| / ||G||): near 0 means the
            # adversarial gradient lies almost entirely inside the clean
            # salient space (Fig. 4.5a behaviour at high l_th).
            print(f"Task 2 Epoch [{epoch+1:03d}/{epochs_task2:03d}] | Val Adv Acc: {val_acc*100:.2f}% "
                  f"| LR: {current_lr:.2e} | ||G_ortho||/||G||: {proj_ratio:.4f}", flush=True)

        if converged:
            print(f" -> Learning rate floor ({MIN_LR:.0e}) reached. Converged at epoch {epoch+1}.", flush=True)
            break

    # Final Benchmark Evaluation
    if verbose:
        print("\n" + "=" * 70, flush=True)
        print(" Final Benchmark Verification", flush=True)
        print("=" * 70, flush=True)

    # Benchmark the checkpoint that was actually selected, not the last epoch.
    # The PGD-AT arm reports a best-of-run number, so reporting last-epoch here
    # would compare best-of-run against a last-epoch snapshot.
    if save_best and os.path.exists(save_name):
        print(f" -> Restoring best-validation checkpoint: {save_name}", flush=True)
        model.load_state_dict(torch.load(save_name, map_location=device))

    final_clean = evaluate(model, final_eval_loader, device, task_id=0, attack_steps=0,
                           desc="Final Clean Head 0", max_batches=max_eval_batches)
    final_pgd20 = evaluate(model, final_eval_loader, device, task_id=1, attack_steps=final_attack_steps,
                           desc=f"Final PGD-{final_attack_steps} Head 1", max_batches=max_eval_batches)
    print(f"Threshold (l_th)      : {threshold}", flush=True)
    print(f"Salient subspace k/n  : {total_k}/{total_n} ({100.0*total_k/total_n:.1f}%)", flush=True)
    print(f"Task 1 Clean Acc      : {task1_clean_acc * 100:.2f}%  (before GPM training)", flush=True)
    print(f"Preserved Clean Acc   : {final_clean * 100:.2f}%  (delta {100*(final_clean-task1_clean_acc):+.2f})", flush=True)
    print(f"PGD-20 Robust Acc     : {final_pgd20 * 100:.2f}%", flush=True)
    print(f"Adversarial epochs    : {epochs_run}", flush=True)
    print(" NOTE: PGD-20 uses head 1 for both attack and defence (task-oracle);", flush=True)
    print("       Sec. 4.3 leaves head combination at test time as future work.", flush=True)

    if no_oracle_eval:
        # OPTIONAL: the same model read without a task oracle.
        #
        # The headline numbers above take clean accuracy from head 0 and robust
        # accuracy from head 1 -- the diagonal of the matrix below. A deployed
        # system cannot pick a head per input, so that pairing is an upper
        # bound rather than an achievable operating point. Each *row* here is a
        # single deployable classifier (g o f) and is therefore directly
        # comparable to the single-head PGD-AT arm.
        #
        # Deliberately not an ensemble: any confidence-routed or averaged
        # combination would need the attack recomputed against the combined
        # function, and confidence routing in particular is the classic
        # gradient-masking failure mode (Athalye et al., obfuscated gradients).
        print("\n" + "-" * 70, flush=True)
        print(" OPTIONAL: no-oracle read (each row is one deployable classifier)", flush=True)
        print("-" * 70, flush=True)

        h1_clean = evaluate(model, final_eval_loader, device, task_id=1, attack_steps=0, desc="Clean Head 1")
        h0_pgd20 = evaluate(model, final_eval_loader, device, task_id=0, attack_steps=20, desc="PGD-20 Head 0")

        print(f"{'':<24s}{'Clean':>10s}{'PGD-20':>10s}", flush=True)
        print(f"{'head 0 (g_clean)':<24s}{final_clean*100:>9.2f}%{h0_pgd20*100:>9.2f}%", flush=True)
        print(f"{'head 1 (g_adv)':<24s}{h1_clean*100:>9.2f}%{final_pgd20*100:>9.2f}%", flush=True)
        print(f"\n Oracle pairing (headline) : clean {final_clean*100:.2f}% / robust {final_pgd20*100:.2f}%", flush=True)
        print(f" Cost of dropping oracle   : {100*(final_clean - h1_clean):+.2f} pts clean, "
              f"if head 1 is deployed", flush=True)
        print(f" Robustness of head 0      : {h0_pgd20*100:.2f}% "
              f"(did the constrained backbone update confer any robustness on the frozen clean head?)",
              flush=True)

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
