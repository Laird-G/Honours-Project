import copy
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from attacks import pgd_attack
from gpm import get_gpm_bases, project_backbone_gradients
from ogp import ReferenceSubspace, reference_gradients, flat_dot, flat_norm

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


def freeze_bn_stats(model):
    """Stop BatchNorm running-stat updates, but leave the affine params trainable.

    The OGP counterpart of freeze_backbone_bn, and the difference is deliberate.
    GPM must also freeze BN *affine* weights because it has no patch-space basis
    to project their gradients against, so they would be an unprojected escape
    hatch. OGP builds its subspace in parameter space and projects EVERY
    trainable tensor, BN affine included -- so freezing them would throw away
    capacity for no protection benefit.

    running_mean / running_var are a different matter in both methods: they are
    buffers updated inside forward(), ignoring requires_grad, the optimizer and
    any projection. Left in train mode during adversarial fine-tuning they track
    the adversarial input distribution, which moves the clean function no matter
    what the gradients do. eval() is the only way to hold them.

    Side benefit: with BN in eval the training-time function equals the deployed
    function, so the reference gradient is exact and the attack is generated
    against the model that will actually be shipped.

    Must be re-applied after every model.train(). Idempotent.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


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
                       objectives=("ce", "kl"), per_class=False, num_classes=10,
                       ref_temp=2.0, project_mode="equality", granularity="global",
                       proj_alpha=1.0, renorm=False, anchor_weight=0.0,
                       select="tradeoff", selection_loader=None, verbose=True):
    """OGP as a single-head adversarial fine-tune of a clean checkpoint.

    Set --ogp_ref ce --ogp_project equality --ogp_granularity global
    --ogp_alpha 1 --ogp_anchor_weight 0 to recover the paper's Algorithm 1;
    every other setting is an extension documented in ogp.py's module docstring.

    Single head throughout, so clean and PGD-20 accuracy come off the SAME
    deployable classifier -- no task oracle, directly comparable to the pgd_at
    arm. That is the structural advantage over train_gpm_pipeline, whose headline
    numbers pair head 0's clean accuracy with head 1's robustness.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 1/2] Loading pre-alignment checkpoint: {clean_checkpoint}", flush=True)
    print("=" * 70, flush=True)
    load_clean_checkpoint(model, clean_checkpoint, device)

    selection_loader = selection_loader if selection_loader is not None else valloader

    # Single head 0 -> one deployable classifier. Head 1 is never in the forward
    # pass, so including it would only put a permanently-None gradient in the
    # parameter list.
    model.base_model.heads[1].requires_grad_(False)
    model.base_model.heads[0].requires_grad_(True)

    # Stats frozen, affine trainable -- see freeze_bn_stats. This closes the one
    # leak a gradient projection provably cannot: running_mean / running_var are
    # buffers, so nothing in eq. 12 can stop them tracking the adversarial
    # distribution.
    freeze_bn_stats(model)

    # theta_0 for the KL reference objective: a frozen copy of the clean model,
    # which is what "general capability" means concretely here.
    teacher = None
    if "kl" in objectives:
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)

    n_dir = (num_classes if per_class else len(ref_loaders)) * len(objectives)
    basis_gb = n_dir * n_param * 4 / 1024 ** 3
    print(f" -> BN stats frozen, BN affine trainable ({len(params)} tensors / "
          f"{n_param:,} parameters, all projected).", flush=True)
    print(f" -> Reference objectives {list(objectives)} x "
          f"{'per-class facets' if per_class else f'{len(ref_loaders)} pools'} "
          f"= up to {n_dir} directions "
          f"(~{basis_gb:.2f} GB basis + the same again transiently per refresh).", flush=True)

    subspace = ReferenceSubspace(
        params, granularity=granularity, delta=delta, mode=project_mode,
        alpha=proj_alpha, renorm=renorm,
    )

    # theta_pre for the optional L2 anchor.
    theta_pre = [p.detach().clone() for p in params] if anchor_weight > 0 else None

    pre_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0, desc="Pre-align Clean")
    print(f" -> Pre-alignment clean accuracy: {pre_clean*100:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"[STAGE 2/2] OGP-Constrained PGD-{adv_steps} Fine-Tune ({epochs} Epochs)", flush=True)
    print(f"    K = {refresh_every} | project = {project_mode} | granularity = {granularity} "
          f"| alpha = {proj_alpha} | renorm = {renorm} | tau = {ref_temp} "
          f"| anchor = {anchor_weight} | delta = {delta}", flush=True)
    print("=" * 70, flush=True)

    # Algorithm 1 line 12 is plain gradient descent, and Appendix A sets weight
    # decay to 0. Both matter here: SGD applies momentum and decay *after* the
    # projection, and theta has a large component along the reference
    # directions, so decay would re-inject exactly what eq. 12 removed. Same
    # reasoning as train_gpm_pipeline's opt_t2.
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

    refreshes, skipped_refreshes = 0, 0
    diagnosed, checked = False, False
    global_step = 0
    best_score, best_epoch = -1.0, 0
    history = []

    for epoch in range(epochs):
        model.train()
        freeze_bn_stats(model)      # model.train() just re-enabled every module

        train_loss, train_correct, train_total = 0.0, 0, 0
        ratio_sum, span_sum, ratio_n = 0.0, 0.0, 0
        active_sum, fallbacks, last_cosines = 0, 0, []

        pbar = tqdm(trainloader, desc=f"OGP [{epoch+1:03d}/{epochs:03d}]", leave=False, dynamic_ncols=True)
        for x, y in pbar:
            # ---- Algorithm 1 lines 3-9: refresh the subspace every K steps ----
            if global_step % refresh_every == 0 and ref_loaders:
                grads, labels = reference_gradients(
                    model, ref_loaders, params, device, task_id=task_id,
                    objectives=objectives, temperature=ref_temp, criterion=criterion,
                    teacher=teacher, per_class=per_class, num_classes=num_classes,
                )
                rank = subspace.build(grads, labels)
                if rank is None:
                    # Non-finite reference gradient. The basis is lagged by
                    # design, so keeping the previous one beats projecting
                    # against garbage.
                    skipped_refreshes += 1
                else:
                    refreshes += 1
                    if not diagnosed:
                        # The single most informative line in the run. Tiny norms
                        # mean the reference direction is mini-batch noise; a
                        # pairwise cosine near 1 means M pools collapsed to one
                        # direction. Both are the failure modes ogp.py's
                        # docstring describes.
                        print(" -> First refresh diagnostics:", flush=True)
                        for lab, n in zip(labels, subspace.ref_norms):
                            print(f"      ||g_ref[{lab}]|| = {n:.4e}", flush=True)
                        for i in range(len(grads)):
                            for j in range(i + 1, len(grads)):
                                d = subspace.ref_norms[i] * subspace.ref_norms[j]
                                if d > 0:
                                    print(f"      cos({labels[i]}, {labels[j]}) = "
                                          f"{flat_dot(grads[i], grads[j]) / d:+.4f}", flush=True)
                        print(f"      retained rank M' = {rank} of {len(grads)} "
                              f"({subspace.dropped_weak} dropped as vanishing, "
                              f"residuals "
                              f"{', '.join(f'{r:.3f}' for r in subspace.residuals)})", flush=True)
                        diagnosed = True
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
            # basis vector from EVERY coordinate, so one overflowed element would
            # be smeared across the whole gradient. Unscaling first also stops
            # scaler.step() from unscaling a second time.
            scaler.unscale_(opt)

            # The anchor gradient goes in BEFORE the projection, not after.
            # lambda*(theta - theta_0) points along the drift the projection is
            # trying to prevent, so adding it afterwards would re-inject exactly
            # the salient-space component eq. 12 just removed -- the same trap as
            # weight decay. Added first, it is projected like any other gradient.
            if anchor_weight > 0:
                with torch.no_grad():
                    for p, p0 in zip(params, theta_pre):
                        p.grad.add_(p.detach() - p0, alpha=anchor_weight)

            if subspace.rank:
                want_stats = (global_step % PROJ_NORM_LOG_EVERY == 0) or not checked
                stats = subspace.project(params, return_stats=want_stats)
                if stats is not None and stats["before"] > 0:
                    ratio_sum += stats["ratio"]
                    span_sum += stats["frac_in_span"]
                    ratio_n += 1
                    active_sum += stats["n_active"]
                    fallbacks += stats["n_fallback"]
                    last_cosines = stats["cosines"]

                if not checked:
                    chk = subspace.selfcheck(
                        params, scale=stats["before"] if stats else None)
                    print(f" -> Self-check ({project_mode}): max|U^T U - I| = "
                          f"{chk['gram_err']:.2e}, {chk['detail']} -> "
                          f"{'PASS' if chk['ok'] else 'WARN'}", flush=True)
                    if max(chk["gram_err"], abs(min(0.0, chk["feas"]))) > 1e-2:
                        raise RuntimeError(
                            f"OGP self-check failed badly ({chk}): the basis is not "
                            f"orthonormal or the projected gradient violates its constraint. "
                            f"Every number this run produces would be meaningless."
                        )
                    if not chk["ok"]:
                        print("    WARNING: above tolerance but below the abort threshold -- "
                              "continuing, but treat the projection as suspect.", flush=True)
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

        # Scored on the selection split, which is disjoint from the reference
        # pools -- otherwise the data that steered the projection would also pick
        # the checkpoint.
        val_clean = evaluate(model, selection_loader, device, task_id=task_id,
                             attack_steps=0, desc="Val Clean")
        val_adv = evaluate(model, selection_loader, device, task_id=task_id,
                           attack_steps=10, desc="Val PGD-10")

        if theta_pre is not None:
            with torch.no_grad():
                drift = math.sqrt(sum(float((p - p0).norm()) ** 2
                                      for p, p0 in zip(params, theta_pre)))
        else:
            drift = float('nan')

        # Selecting on robustness alone would pick the most-forgetting epoch,
        # which is the opposite of what this arm is for.
        saved = ""
        if select == "none":
            # The paper reports the final model after a fixed budget; the
            # end-of-run _last.pth write covers it, so don't churn 146 MB/epoch.
            best_epoch = epoch + 1
        else:
            score = {"tradeoff": 0.5 * (val_clean + val_adv),
                     "robust": val_adv,
                     "clean": val_clean}[select]
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                torch.save(model.state_dict(), save_name)
                saved = " -> saved"

        proj_ratio = (ratio_sum / ratio_n) if ratio_n else float('nan')
        frac_span = (span_sum / ratio_n) if ratio_n else float('nan')
        history.append({"epoch": epoch + 1, "val_clean": val_clean, "val_robust": val_adv,
                        "ratio": proj_ratio, "frac_in_span": frac_span, "drift": drift})

        if verbose:
            cos_str = ", ".join(f"{c:+.3f}" for c in last_cosines[:4]) if last_cosines else "n/a"
            extra = f" | GEM active/step: {active_sum/max(1,ratio_n):.1f}" \
                    if project_mode == "inequality" else ""
            if fallbacks:
                extra += f" | QP fallbacks: {fallbacks}"
            print(f"OGP Epoch [{epoch+1:03d}/{epochs:03d}] | Val Clean: {val_clean*100:.2f}% "
                  f"| Val PGD-10: {val_adv*100:.2f}% | LR: {opt.param_groups[0]['lr']:.2e} "
                  f"| ||g'||/||g||: {proj_ratio:.4f} | in-span: {frac_span:.4f} "
                  f"| M'={subspace.rank} | drift: {drift:.3f} | cos: [{cos_str}]{extra}{saved}",
                  flush=True)

    final_name = save_name.replace(".pth", "_last.pth")
    torch.save(model.state_dict(), final_name)

    print("\n" + "=" * 70, flush=True)
    print(" Final Benchmark Verification (single head, no task oracle)", flush=True)
    print("=" * 70, flush=True)
    if select != "none" and os.path.exists(save_name):
        print(f" -> Benchmarking the selected checkpoint (epoch {best_epoch}, "
              f"'{select}' criterion); last epoch kept at {final_name}", flush=True)
        model.load_state_dict(torch.load(save_name, map_location=device))

    final_clean = evaluate(model, testloader, device, task_id=task_id, attack_steps=0,
                           desc="Final Clean")
    final_pgd20 = evaluate(model, testloader, device, task_id=task_id, attack_steps=20,
                           desc="Final PGD-20")

    print(f"Projection            : {project_mode} / {granularity} / alpha={proj_alpha}"
          f"{' / renorm' if renorm else ''}", flush=True)
    print(f"Reference subspace    : {list(objectives)}"
          f"{' per-class' if per_class else f' x {len(ref_loaders)} pools'}, "
          f"rank M' = {subspace.rank}, K = {refresh_every}", flush=True)
    print(f"Refreshes / skipped   : {refreshes} / {skipped_refreshes}", flush=True)
    print(f"Selected epoch        : {best_epoch} of {epochs} ({select})", flush=True)
    print(f"Pre-align Clean Acc   : {pre_clean*100:.2f}%", flush=True)
    print(f"Preserved Clean Acc   : {final_clean*100:.2f}%  "
          f"(delta {100*(final_clean-pre_clean):+.2f})", flush=True)
    print(f"PGD-20 Robust Acc     : {final_pgd20*100:.2f}%", flush=True)
    print(" NOTE: both numbers come from the same head, so this row is ONE", flush=True)
    print("       deployable classifier -- directly comparable to the pgd_at arm.", flush=True)

    return {
        "refresh_every": refresh_every,
        "objectives": list(objectives),
        "project_mode": project_mode,
        "granularity": granularity,
        "alpha": proj_alpha,
        "rank": subspace.rank,
        "pre_clean": pre_clean,
        "final_clean": final_clean,
        "final_robust": final_pgd20,
        "best_epoch": best_epoch,
        "epochs_run": epochs,
        "history": history,
    }