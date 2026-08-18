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

# Sec. 4.4.2: "training process continues until learning rate reaches a
# minimum value of 1e-7". ReduceLROnPlateau clamps *at* min_lr, so the
# stopping test has to be <=, not <.
MIN_LR = 1e-7
# How often to record ||G|| before/after projection (Figs. 4.5 / 4.7).
PROJ_NORM_LOG_EVERY = 50


def load_clean_checkpoint(model, path, device):
    """Load a Task 1 / pre-alignment checkpoint, refusing partial matches.

    strict=False is needed because a checkpoint may legitimately predate a head
    being added, but it would otherwise let a key mismatch through silently and
    run an entire sweep on a randomly initialised backbone, producing
    plausible-looking but meaningless curves.
    """
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


def train_ogp_pipeline(model, trainloader, valloader, testloader, criterion, ref_loaders,
                       clean_checkpoint, epochs, lr, refresh_every, delta, warmup_ratio,
                       adv_alpha, adv_steps, device, save_name, task_id=0, verbose=True):
    """OGPSA (OGP.pdf Algorithm 1) as a single-head adversarial fine-tune.

    Faithful port. The mapping is: theta_pre = the clean checkpoint,
    L_safe = adversarial cross-entropy, D_ref = small fixed pools of clean
    images, L_ref = clean cross-entropy. One head throughout, so clean and
    PGD-20 accuracy are read off the *same* deployable classifier.

    Deliberately absent, matching the paper: no early stopping, no best-model
    selection (Appendix A trains a fixed 1-3 epochs and reports the final
    model), no renormalisation of the projected gradient (eq. 13), and no
    special handling of BatchNorm running statistics -- see the known-weaknesses
    section of the plan; they are buffers, not parameters, so the projection
    cannot reach them and they will track the adversarial distribution.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 1/2] Loading pre-alignment checkpoint: {clean_checkpoint}", flush=True)
    print("=" * 70, flush=True)
    load_clean_checkpoint(model, clean_checkpoint, device)

    # Single-head: head 1 is never used in the forward pass, so it would only
    # sit in the parameter list with a permanently None gradient.
    model.base_model.heads[1].requires_grad_(False)
    model.base_model.heads[0].requires_grad_(True)

    # Fixed order, built once. The basis vectors are stored in this same layout,
    # so the flat inner products in ogp.py line up element for element.
    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f" -> Projecting over {len(params)} tensors / {n_param:,} parameters "
          f"(one global subspace, not per layer).", flush=True)

    pre_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0,
                        desc="Pre-align Clean")
    print(f" -> Pre-alignment clean accuracy: {pre_clean*100:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 2/2] OGP-Constrained PGD-{adv_steps} Fine-Tune ({epochs} Epochs, "
          f"K = {refresh_every}, M = {len(ref_loaders)})", flush=True)
    print("=" * 70, flush=True)

    # Algorithm 1 line 12 is plain gradient descent, and Appendix A sets weight
    # decay to 0. Both matter here: SGD applies momentum and decay *after* the
    # projection, and theta has a large component along the reference
    # directions, so decay would re-inject exactly what eq. 12 removed. Same
    # reasoning as train_gpm_pipeline's opt_t2.
    opt = optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)

    # Appendix A: cosine schedule with a 0.1 warm-up ratio. Per *step*, not per
    # epoch, because the warm-up is a fraction of total optimisation steps.
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
        train_loss, train_correct, train_total = 0.0, 0, 0
        ratio_sum, ratio_n = 0.0, 0
        last_cosines = []

        pbar = tqdm(trainloader, desc=f"OGP [{epoch+1:03d}/{epochs:03d}]", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            # ---- Algorithm 1 lines 3-9: refresh the subspace every K steps ----
            if global_step % refresh_every == 0:
                grads = reference_gradients(model, ref_loaders, criterion, params, device,
                                            task_id=task_id)
                ref_norms = [flat_norm(g) for g in grads]

                if all(math.isfinite(n) for n in ref_norms):
                    if not diagnosed:
                        # Is the reference direction real signal or just
                        # mini-batch noise? A converged clean model has near-zero
                        # loss, and a noise direction has cosine ~1/sqrt(d) with
                        # anything, i.e. no constraint at all.
                        print(f" -> First refresh: ||g_ref|| = "
                              f"{', '.join(f'{n:.3e}' for n in ref_norms)}", flush=True)
                        for i in range(len(grads)):
                            for j in range(i + 1, len(grads)):
                                denom = ref_norms[i] * ref_norms[j]
                                cij = flat_dot(grads[i], grads[j]) / denom if denom > 0 else 0.0
                                print(f"    cos(g_ref[{i}], g_ref[{j}]) = {cij:+.4f}", flush=True)
                        diagnosed = True

                    basis, residuals = gram_schmidt(grads, delta=delta)
                    refreshes += 1
                else:
                    # The basis is lagged by design, so keeping the previous one
                    # is strictly better than projecting against garbage.
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
            # Unscale before projecting: the projection subtracts a multiple of a
            # basis vector from *every* coordinate, so one overflowed element
            # would be smeared across the whole gradient. Unscaling first also
            # stops scaler.step() from unscaling a second time.
            scaler.unscale_(opt)

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
                    # Order-of-magnitude failures mean the projection is wrong
                    # (mismatched flatten order, stale basis, wrong parameter
                    # list) and every number the run produces would be
                    # meaningless. Anything smaller is fp32 reduction noise over
                    # 3.7e7 elements: worth flagging, not worth aborting for.
                    if max(gram_err, resid) > 1e-2:
                        raise RuntimeError(
                            f"OGP self-check failed badly (max error "
                            f"{max(gram_err, resid):.2e}): the basis is not orthonormal or "
                            f"the projected gradient is not orthogonal to it. Refusing to train."
                        )
                    if not ok:
                        print("    WARNING: above the 1e-4 tolerance but below the abort "
                              "threshold -- continuing, but treat the projection as suspect.",
                              flush=True)
                    checked = True

            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1

            train_loss += loss.item() * y.size(0)
            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)
            pbar.set_postfix(adv_loss=f"{train_loss/train_total:.3f}",
                             adv_acc=f"{100*train_correct/train_total:.1f}%")

        val_clean = evaluate(model, valloader, device, task_id=task_id, attack_steps=0,
                             desc="Val Clean")
        val_adv = evaluate(model, valloader, device, task_id=task_id, attack_steps=10,
                           desc="Val PGD-10")

        if verbose:
            # proj_ratio is ||g'||/||g||: 1.0 means the projection is doing
            # nothing (the adversarial gradient is already orthogonal to the
            # clean reference directions), 0.0 means it lies entirely inside them.
            proj_ratio = (ratio_sum / ratio_n) if ratio_n else float('nan')
            cos_str = ", ".join(f"{c:+.3f}" for c in last_cosines) if last_cosines else "n/a"
            print(f"OGP Epoch [{epoch+1:03d}/{epochs:03d}] | Val Clean: {val_clean*100:.2f}% "
                  f"| Val PGD-10: {val_adv*100:.2f}% | LR: {opt.param_groups[0]['lr']:.2e} "
                  f"| ||g'||/||g||: {proj_ratio:.4f} | rank M'={len(basis)} "
                  f"| cos(g, u): {cos_str}", flush=True)

    # No best-model selection: Appendix A reports the final model after a fixed
    # budget, so selecting here would not be the published method.
    torch.save(model.state_dict(), save_name)
    print(f" -> Saved final model to {save_name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" Final Benchmark Verification (single head, no task oracle)", flush=True)
    print("=" * 70, flush=True)
    final_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0,
                           desc="Final Clean")
    final_pgd20 = evaluate(model, testloader, device, task_id=task_id, attack_steps=20,
                           desc="Final PGD-20")

    print(f"Refresh period K      : {refresh_every}", flush=True)
    print(f"Reference sets M      : {len(ref_loaders)}  (retained rank M' = {len(basis)}, "
          f"residuals {', '.join(f'{r:.3f}' for r in residuals) if residuals else 'n/a'})", flush=True)
    print(f"Refreshes / skipped   : {refreshes} / {skipped_refreshes}", flush=True)
    print(f"Pre-align Clean Acc   : {pre_clean*100:.2f}%", flush=True)
    print(f"Preserved Clean Acc   : {final_clean*100:.2f}%  "
          f"(delta {100*(final_clean-pre_clean):+.2f})", flush=True)
    print(f"PGD-20 Robust Acc     : {final_pgd20*100:.2f}%", flush=True)
    print(f"Adversarial epochs    : {epochs}", flush=True)
    print(" NOTE: both numbers come from the same head, so this row is one", flush=True)
    print("       deployable classifier and is directly comparable to pgd_at.", flush=True)

    return {
        "refresh_every": refresh_every,
        "num_refs": len(ref_loaders),
        "rank": len(basis),
        "pre_clean": pre_clean,
        "final_clean": final_clean,
        "final_robust": final_pgd20,
        "epochs_run": epochs,
    }
