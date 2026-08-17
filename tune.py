"""PyHopper hyperparameter search for the clean / pgd_at / gpm arms.

Design notes
------------
* Only *optimisation* hyperparameters are searched. Budget knobs (epochs,
  epochs_task1/2), the threat model (eps = 8/255), the evaluation attack
  (PGD-20), the architecture (WRN-28-10) and the dataset are held fixed --
  searching them would change what is being compared rather than how well it
  is trained, and the comparison of PGD-AT against GPM is the point.
* GPM Task 2 momentum and weight decay are also held at 0 on purpose. SGD
  applies both *after* the projection, so a non-zero value re-injects exactly
  the salient-space component the projection removed (Sec. 4.4.2). They are
  method constraints, not free parameters.
* Each trial is a short *proxy* run (--proxy_epochs), scored on the 5% held-out
  validation split -- never the test set, which stays untouched for the final
  numbers. Proxy ranking assumes the ordering of configurations is roughly
  preserved at full budget; re-run the top configuration at full budget through
  main.py before reporting anything.
* Every trial rebuilds the model from the same seed, so trials differ only in
  their hyperparameters.

Examples
--------
  python tune.py --mode clean   --steps 20 --proxy_epochs 15
  python tune.py --mode pgd_at  --runtime 8h --proxy_epochs 10
  python tune.py --mode gpm     --steps 30 --proxy_epochs 8 \
                 --clean_checkpoint task1_clean_backbone.pth
"""

import argparse
import copy
import json
import os
import random

import pyhopper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import get_dataloaders
from models import WideResNet, NormalizedModel
from methods import METHODS
from trainer import train_standard, train_gpm_pipeline

torch.backends.cudnn.benchmark = True

DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

# Loaders are cached by batch size: each DataLoader here holds persistent
# workers, and building a fresh set per trial would leak worker processes for
# the length of the search.
_LOADER_CACHE = {}


def get_loaders(args, batch_size):
    key = (args.dataset, batch_size, args.proxy_train_frac)
    if key not in _LOADER_CACHE:
        trainloader, valloader, testloader, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=batch_size,
            num_workers=args.num_workers, return_val=True,
        )
        if args.proxy_train_frac < 1.0:
            # A fixed random subset of the training split, so a trial costs a
            # fraction of an epoch's compute and more configurations fit in the
            # budget. Deterministic (seed 42, as in dataset.py) and identical
            # across trials, so trials stay comparable. This trades absolute
            # accuracy for throughput -- the *ranking* is what the search needs.
            base = trainloader.dataset
            n = int(len(base) * args.proxy_train_frac)
            idx = torch.randperm(
                len(base), generator=torch.Generator().manual_seed(42)
            ).tolist()[:n]
            trainloader = DataLoader(
                Subset(base, idx), batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                persistent_workers=args.num_workers > 0,
            )
        _LOADER_CACHE[key] = (trainloader, valloader, testloader, num_classes)
    return _LOADER_CACHE[key]


def fresh_model(args, num_classes, device):
    """Identical initialisation for every trial, so trials are comparable."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mean, std = DATASET_STATS[args.dataset]
    base = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=10, num_tasks=2)
    return NormalizedModel(base, mean=mean, std=std).to(device)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def build_space(mode):
    """Bounds are centred on the current defaults so the search starts from a
    known-good region rather than exploring somewhere the paper's schedule
    cannot be recovered from."""
    if mode == "clean":
        return pyhopper.Search(
            lr=pyhopper.float(0.02, 0.4, log=True, precision=2, init=0.1),
            momentum=pyhopper.float(0.8, 0.98, precision=2, init=0.9),
            weight_decay=pyhopper.float(1e-4, 2e-3, log=True, precision=1, init=5e-4),
            # pyhopper.bool()'s init argument is an index, not a value; spell the
            # options out so "start from nesterov=False" is unambiguous.
            nesterov=pyhopper.choice([False, True], init_index=0),
            label_smoothing=pyhopper.float(0.0, 0.2, precision=2, init=0.0),
            # init_index=1 -> 256, the current default. Ordinal so mutation
            # prefers the neighbouring batch size.
            batch_size=pyhopper.choice([128, 256, 512], init_index=1, is_ordinal=True),
        )
    if mode == "pgd_at":
        return pyhopper.Search(
            lr=pyhopper.float(0.02, 0.4, log=True, precision=2, init=0.1),
            momentum=pyhopper.float(0.8, 0.98, precision=2, init=0.9),
            weight_decay=pyhopper.float(1e-4, 2e-3, log=True, precision=1, init=5e-4),
            # pyhopper.bool()'s init argument is an index, not a value; spell the
            # options out so "start from nesterov=False" is unambiguous.
            nesterov=pyhopper.choice([False, True], init_index=0),
            label_smoothing=pyhopper.float(0.0, 0.2, precision=2, init=0.0),
            batch_size=pyhopper.choice([128, 256, 512], init_index=1, is_ordinal=True),
            # Inner-maximisation solver. eps is fixed; alpha is expressed in
            # /255 units and steps trades attack quality against wall clock.
            train_alpha_255=pyhopper.float(1.0, 4.0, precision=1, init=2.0),
            train_steps=pyhopper.int(5, 10, init=7),
        )
    if mode == "gpm":
        return pyhopper.Search(
            threshold=pyhopper.float(0.80, 0.99, precision=2, init=0.95),
            lr_task2=pyhopper.float(1e-3, 1e-1, log=True, precision=2, init=0.01),
            plateau_factor=pyhopper.float(0.2, 0.6, precision=2, init=0.33),
            plateau_patience=pyhopper.int(2, 8, init=5),
            gpm_alpha_255=pyhopper.float(1.0, 4.0, precision=1, init=2.0),
            gpm_steps=pyhopper.int(7, 10, init=10),
            # None = use the whole val split. Fewer samples flatten the spectrum
            # less and so shrink k for a given l_th (see gpm.py); ordered
            # ascending with None last so the ordinal mutation is meaningful.
            gpm_samples=pyhopper.choice([128, 512, 2000, None], init_index=3, is_ordinal=True),
            batch_size=pyhopper.choice([128, 256, 512], init_index=1, is_ordinal=True),
        )
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def objective_standard(params, args, device, mode):
    trainloader, valloader, _testloader, num_classes = get_loaders(args, params["batch_size"])
    model = fresh_model(args, num_classes, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"],
                          weight_decay=params["weight_decay"], nesterov=params["nesterov"])
    # The proxy run has to anneal over its own horizon; reusing the 200-epoch
    # cosine (or the [100, 125] milestones) would leave every trial at its
    # initial learning rate and score the warm-up phase instead of the run.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.proxy_epochs)

    step_kwargs = None
    if mode == "pgd_at":
        step_kwargs = {"alpha": params["train_alpha_255"] / 255.0, "steps": params["train_steps"]}

    # Scored on the validation split, evaluated only near the end of the proxy
    # run -- a PGD-20 pass is expensive and only the final value is used.
    return train_standard(
        model=model, trainloader=trainloader, testloader=valloader,
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        step_fn=METHODS[mode], mode=mode, epochs=args.proxy_epochs, device=device,
        save_name="tune_tmp.pth",        # unused: save_best=False
        step_kwargs=step_kwargs,
        eval_every=max(1, args.proxy_epochs // 2),
        max_eval_batches=args.max_eval_batches,
        save_best=False,
        verbose=args.verbose_trials,
    )


def objective_gpm(params, args, device):
    trainloader, valloader, _testloader, num_classes = get_loaders(args, params["batch_size"])
    model = fresh_model(args, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    result = train_gpm_pipeline(
        model=model, trainloader=trainloader, valloader=valloader, testloader=valloader,
        criterion=criterion,
        threshold=params["threshold"],
        clean_checkpoint=args.clean_checkpoint,
        epochs_task1=0,                      # unreachable: a checkpoint is required
        epochs_task2=args.proxy_epochs,
        device=device,
        save_name="tune_gpm_tmp.pth",
        gpm_samples=params["gpm_samples"],
        lr_task2=params["lr_task2"],
        plateau_factor=params["plateau_factor"],
        plateau_patience=params["plateau_patience"],
        adv_alpha=params["gpm_alpha_255"] / 255.0,
        adv_steps=params["gpm_steps"],
        save_best=False,                     # no checkpoint churn per trial
        verbose=args.verbose_trials,
        final_eval_loader=valloader,         # test set stays untouched
        final_attack_steps=10,               # PGD-20 is for the final report only
        max_eval_batches=args.max_eval_batches,
    )

    # GPM has two competing objectives: retain head-0 clean accuracy and gain
    # head-1 robustness. A pure-robustness score would happily pick l_th -> 0,
    # which is just unconstrained adversarial training and answers nothing about
    # the method. Retention is measured as a *drop* against the Task 1 reference
    # so a weak Task 1 checkpoint cannot be gamed by scoring absolute clean
    # accuracy.
    clean_drop = max(0.0, result["task1_clean"] - result["final_clean"])
    score = result["final_robust"] - args.clean_penalty * clean_drop

    print(f"    l_th={params['threshold']:.2f} k/n={result['k']}/{result['n']} "
          f"clean {result['final_clean']*100:.2f}% (drop {clean_drop*100:+.2f}) "
          f"robust {result['final_robust']*100:.2f}% -> score {score:.4f}", flush=True)
    return score


def main():
    p = argparse.ArgumentParser(description="PyHopper hyperparameter search")
    p.add_argument("--mode", type=str, required=True, choices=["clean", "pgd_at", "gpm"])
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--depth", type=int, default=28)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--proxy_epochs", type=int, default=15,
                   help="Epochs per trial. NOT searched -- it is the search budget, not a "
                        "hyperparameter. Raise it if the proxy ranking looks unstable.")
    p.add_argument("--proxy_train_frac", type=float, default=1.0,
                   help="Fraction of the training split used per trial (fixed subset). "
                        "0.5 halves trial cost so roughly twice as many configurations fit "
                        "in the same wall clock.")
    p.add_argument("--max_eval_batches", type=int, default=None,
                   help="Truncate each in-search evaluation to N batches (cheaper, noisier)")
    p.add_argument("--steps", type=int, default=None, help="Number of trials")
    p.add_argument("--runtime", type=str, default=None, help='Wall-clock budget, e.g. "6h 30min"')
    p.add_argument("--clean_penalty", type=float, default=1.0,
                   help="gpm only: weight on head-0 clean-accuracy drop in the objective. "
                        "0 = pure robustness, 1 = one point of forgetting cancels one point "
                        "of robustness.")
    p.add_argument("--clean_checkpoint", type=str, default=None,
                   help="gpm only, required: Task 1 backbone. Re-training Task 1 inside every "
                        "trial would make the search unaffordable and would confound Task 1 "
                        "quality with the Task 2 hyperparameters under test.")
    p.add_argument("--out", type=str, default=None, help="Where to write the best params (JSON)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="PyHopper checkpoint file; an existing one is resumed")
    p.add_argument("--verbose_trials", action="store_true", help="Full per-epoch logs inside trials")
    args = p.parse_args()

    if args.steps is None and args.runtime is None:
        args.steps = 20
    if args.mode == "gpm":
        if not args.clean_checkpoint or not os.path.exists(args.clean_checkpoint):
            p.error("--mode gpm requires an existing --clean_checkpoint (e.g. task1_clean_backbone.pth)")
    if not torch.cuda.is_available():
        # autocast('cuda'), GradScaler and the PGD loss scaling all assume CUDA.
        p.error("No CUDA device visible; the training code paths assume a GPU.")

    device = torch.device("cuda")
    out_path = args.out or f"best_params_{args.mode}_{args.dataset}.json"

    search = build_space(args.mode)

    def objective(params):
        if args.mode == "gpm":
            return objective_gpm(params, args, device)
        return objective_standard(params, args, device, args.mode)

    metric = {"clean": "val clean acc", "pgd_at": "val PGD-20 acc",
              "gpm": f"val PGD-10 acc - {args.clean_penalty} x clean drop"}[args.mode]
    print(f"Searching {args.mode} on {args.dataset} | objective: maximise {metric} "
          f"| {args.proxy_epochs} proxy epochs/trial "
          f"| {100*args.proxy_train_frac:.0f}% of the train split", flush=True)

    best = search.run(
        objective, direction="maximize",
        steps=args.steps, runtime=args.runtime,
        n_jobs=1,                            # one model at a time: WRN-28-10 fills the GPU
        checkpoint_path=args.checkpoint,
        ignore_nans=False,
    )

    best = copy.deepcopy(best)
    # Report the attack step size in both unit systems: the search works in /255
    # so that mutations stay on a sane scale, main.py takes the raw float.
    for k in ("train_alpha_255", "gpm_alpha_255"):
        if k in best:
            best[k.replace("_255", "")] = best[k] / 255.0

    payload = {"mode": args.mode, "dataset": args.dataset, "objective": metric,
               "best_value": float(search.best_f), "proxy_epochs": args.proxy_epochs,
               "params": best}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n" + "=" * 70, flush=True)
    print(f"Best {metric}: {search.best_f:.4f}  (proxy run, {args.proxy_epochs} epochs)", flush=True)
    for k, v in best.items():
        print(f"  {k:<18s} {v}", flush=True)
    print(f"\nWritten to {out_path}", flush=True)
    print("\nFull-budget re-run:\n  " + full_run_command(args, best), flush=True)
    print("\n(The proxy score is a ranking signal only -- the number to report comes "
          "from the full-budget run above, on the test set.)", flush=True)


def full_run_command(args, best):
    cmd = [f"python main.py --mode {args.mode} --dataset {args.dataset}"]
    if "batch_size" in best:
        cmd.append(f"--batch_size {best['batch_size']}")
    if args.mode in ("clean", "pgd_at"):
        cmd.append(f"--lr {best['lr']} --momentum {best['momentum']} "
                   f"--weight_decay {best['weight_decay']} "
                   f"--label_smoothing {best['label_smoothing']}")
        if best.get("nesterov"):
            cmd.append("--nesterov")
    if args.mode == "pgd_at":
        cmd.append(f"--train_alpha {best['train_alpha']} --train_steps {best['train_steps']}")
    if args.mode == "gpm":
        cmd.append(f"--threshold {best['threshold']} --lr_task2 {best['lr_task2']} "
                   f"--plateau_factor {best['plateau_factor']} "
                   f"--plateau_patience {best['plateau_patience']} "
                   f"--gpm_alpha {best['gpm_alpha']} --gpm_steps {best['gpm_steps']}")
        if best.get("gpm_samples") is not None:
            cmd.append(f"--gpm_samples {best['gpm_samples']}")
        if args.clean_checkpoint:
            cmd.append(f"--clean_checkpoint {args.clean_checkpoint}")
    return " ".join(cmd)


if __name__ == "__main__":
    main()
