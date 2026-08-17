import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from models import WideResNet, NormalizedModel
from methods import METHODS
from trainer import train_standard, train_gpm_pipeline

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
    parser.add_argument("--mode", type=str, default="clean", choices=["clean", "pgd_at", "gpm"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epochs for standard clean/pgd_at training (default: 150 clean, 200 pgd_at)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=28,
                        help="28 = WRN-28-10; 34 = Madry's 5-residual-units-per-group variant")

    # GPM-specific parameters
    parser.add_argument("--threshold", type=float, default=0.95, help="GPM energy retention threshold (0.1 to 0.99)")
    parser.add_argument("--clean_checkpoint", type=str, default=None, help="Path to pre-trained clean checkpoint to skip Stage 1")
    parser.add_argument("--epochs_task1", type=int, default=150, help="Clean training epochs if no checkpoint provided")
    parser.add_argument("--epochs_task2", type=int, default=200,
                        help="Cap on GPM Task 2 epochs. Matched to --epochs so the cap is not the "
                             "binding constraint and the Sec. 4.4.2 plateau rule decides when to stop; "
                             "keeps the adversarial-epoch budget comparable to the pgd_at arm.")
    parser.add_argument("--gpm_samples", type=int, default=None,
                        help="Cap images used for GPM basis extraction (GPM uses ~1e2; "
                             "more samples flatten the spectrum and inflate k for a given l_th)")
    parser.add_argument("--no_oracle_eval", action="store_true",
                        help="Also report the full head x {clean, PGD-20} matrix. The headline "
                             "numbers pair head-0 clean with head-1 robust, which assumes a task "
                             "oracle at test time; each row of the matrix is a single deployable "
                             "classifier and is comparable to the pgd_at arm. Adds one clean eval "
                             "and one full PGD-20 eval over the test set.")

    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 150 if args.mode == "clean" else 200

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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # Clean training follows Sharmin 2022 Sec. 4.4.2 (lr 0.1, x0.1 at 100
        # and 125) so that a checkpoint produced here is a valid Task 1 model
        # for the GPM pipeline. pgd_at keeps cosine annealing.
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
            save_name=f"best_{args.mode}_{args.dataset}.pth"
        )

    elif args.mode == "gpm":
        trainloader, valloader, testloader, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, return_val=True
        )

        base_model = WideResNet(depth=args.depth, num_classes=num_classes, widen_factor=10, num_tasks=2)
        model = NormalizedModel(base_model, mean=mean, std=std).to(device)
        criterion = nn.CrossEntropyLoss()

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
        )


if __name__ == "__main__":
    main()
