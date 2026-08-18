import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from models import WideResNet, NormalizedModel
from methods import METHODS
from ogp import make_reference_loaders
from trainer import train_standard, train_gpm_pipeline, train_ogp_pipeline

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark stays on: algorithm selection introduces tiny float
    # nondeterminism but cannot change which basis vectors are selected. The
    # determinism that matters for an l_th sweep -- a fixed, un-augmented
    # extraction set -- is handled in dataset.get_dataloaders.


def main():
    parser = argparse.ArgumentParser(description="Adversarial & Continual Learning Framework")
    parser.add_argument("--mode", type=str, default="clean", choices=["clean", "pgd_at", "gpm", "ogp"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs for standard clean/pgd_at training (default: 150 clean, 200 pgd_at)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--train_alpha", type=float, default=2/255,
                        help="PGD-AT training step size (eps stays 8/255: threat model, not a hyperparameter)")
    parser.add_argument("--train_steps", type=int, default=7,
                        help="PGD-AT training attack steps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=28,
                        help="28 = WRN-28-10; 34 = Madry's 5-residual-units-per-group variant")

    # ---------------------------------------------------------
    # GPM-specific parameters
    # ---------------------------------------------------------
    parser.add_argument("--threshold", type=float, default=0.95, help="GPM energy retention threshold (0.1 to 0.99)")
    parser.add_argument("--clean_checkpoint", type=str, default=None, help="Path to pre-trained clean checkpoint to skip Stage 1")
    parser.add_argument("--epochs_task1", type=int, default=150, help="Clean training epochs if no checkpoint provided")
    parser.add_argument("--epochs_task2", type=int, default=200,
                        help="Cap on GPM Task 2 epochs. Matched to --epochs so the cap is not the "
                             "binding constraint and the Sec. 4.4.2 plateau rule decides when to stop; "
                             "keeps the adversarial-epoch budget comparable to the pgd_at arm.")
    parser.add_argument("--lr_task2", type=float, default=0.01, help="GPM Task 2 SGD learning rate")
    parser.add_argument("--plateau_factor", type=float, default=1/3,
                        help="ReduceLROnPlateau decay factor for GPM Task 2")
    parser.add_argument("--plateau_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience for GPM Task 2")
    parser.add_argument("--gpm_alpha", type=float, default=2/255, help="GPM Task 2 training PGD step size")
    parser.add_argument("--gpm_steps", type=int, default=10, help="GPM Task 2 training PGD steps")
    parser.add_argument("--gpm_samples", type=int, default=None,
                        help="Cap images used for GPM basis extraction (GPM uses ~1e2; "
                             "more samples flatten the spectrum and inflate k for a given l_th)")

    # ---------------------------------------------------------
    # OGP-specific parameters (OGPSA Enhanced Vision Port)
    # ---------------------------------------------------------
    parser.add_argument("--epochs_ogp", type=int, default=30,
                        help="Adversarial fine-tune epochs for --mode ogp")
    parser.add_argument("--ogp_lr", type=float, default=0.01, help="OGP fine-tune learning rate")
    parser.add_argument("--ogp_refresh", type=int, default=30,
                        help="K: subspace refresh period in steps (paper: 30 for SFT, 5 for DPO)")
    parser.add_argument("--ogp_num_refs", type=int, default=8,
                        help="M: number of reference pools (increased from 2 to 8 for rich class coverage)")
    parser.add_argument("--ogp_ref_samples", type=int, default=256,
                        help="Images per reference set D_ref^(i)")
    parser.add_argument("--ogp_ref_batch", type=int, default=64,
                        help="Batch B^(i) drawn from each reference set per refresh")
    parser.add_argument("--ogp_delta", type=float, default=0.05,
                        help="Gram-Schmidt collinearity threshold (eq. 11)")
    parser.add_argument("--ogp_warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps spent in linear LR warm-up")
    parser.add_argument("--ogp_anchor_weight", type=float, default=0.01,
                        help="L2 weight anchor lambda toward theta_pre to guard against Hessian curvature drift")
    parser.add_argument("--ogp_ref_temp", type=float, default=2.0,
                        help="Temperature scaling tau for reference gradients to prevent zero-gradient collapse")

    parser.add_argument("--no_oracle_eval", action="store_true",
                        help="Also report the full head x {clean, PGD-20} matrix. The headline "
                             "numbers pair head-0 clean with head-1 robust, which assumes a task "
                             "oracle at test time; each row of the matrix is a single deployable "
                             "classifier and is comparable to the pgd_at arm. Adds one clean eval "
                             "and one full PGD-20 eval over the test set.")

    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 150 if args.mode == "clean" else 200
    if args.mode == "ogp" and not (args.clean_checkpoint and os.path.exists(args.clean_checkpoint)):
        parser.error("--mode ogp requires an existing --clean_checkpoint (e.g. best_clean_cifar10.pth)")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if args.dataset == "cifar100" \
        else ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if args.mode in ["clean", "pgd_at"]:
        trainloader, testloader, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, return_val=False
        )

        base_model = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=10, num_tasks=2)
        model = NormalizedModel(base_model, mean=mean, std=std).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
        scheduler = (optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 125], gamma=0.1)
                     if args.mode == "clean"
                     else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs))

        train_standard(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            step_fn=METHODS[args.mode],
            mode=args.mode,
            epochs=args.epochs,
            device=device,
            save_name=f"best_{args.mode}_{args.dataset}.pth",
            step_kwargs=({"alpha": args.train_alpha, "steps": args.train_steps}
                         if args.mode == "pgd_at" else None),
        )

    elif args.mode == "gpm":
        trainloader, valloader, testloader, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, return_val=True
        )

        base_model = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=10, num_tasks=2)
        model = NormalizedModel(base_model, mean=mean, std=std).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        train_gpm_pipeline(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            criterion=criterion,
            threshold=args.threshold,
            clean_checkpoint=args.clean_checkpoint,
            epochs_task1=args.epochs_task1,
            epochs_task2=args.epochs_task2,
            device=device,
            save_name=f"best_gpm_th{args.threshold}_{args.dataset}.pth",
            gpm_samples=args.gpm_samples,
            no_oracle_eval=args.no_oracle_eval,
            lr_task1=args.lr,
            lr_task2=args.lr_task2,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            adv_alpha=args.gpm_alpha,
            adv_steps=args.gpm_steps,
        )

    elif args.mode == "ogp":
        trainloader, valloader, testloader, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, return_val=True
        )

        base_model = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=10, num_tasks=2)
        model = NormalizedModel(base_model, mean=mean, std=std).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Build diverse reference pools from clean un-augmented validation data
        ref_loaders = make_reference_loaders(
            valloader,
            num_refs=args.ogp_num_refs,
            ref_samples=args.ogp_ref_samples,
            ref_batch=args.ogp_ref_batch,
            seed=args.seed + 1234,
        )

        train_ogp_pipeline(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            criterion=criterion,
            ref_loaders=ref_loaders,
            clean_checkpoint=args.clean_checkpoint,
            epochs=args.epochs_ogp,
            lr=args.ogp_lr,
            refresh_every=args.ogp_refresh,
            delta=args.ogp_delta,
            warmup_ratio=args.ogp_warmup_ratio,
            adv_alpha=args.train_alpha,
            adv_steps=args.train_steps,
            device=device,
            save_name=f"final_ogp_K{args.ogp_refresh}_M{args.ogp_num_refs}_{args.dataset}.pth",
            task_id=0,
            anchor_weight=args.ogp_anchor_weight,
            ref_temp=args.ogp_ref_temp,
            verbose=True,
        )


if __name__ == "__main__":
    main()