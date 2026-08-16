import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from models import WideResNet, NormalizedModel
from methods import METHODS
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pgd_at", choices=list(METHODS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader, num_classes = get_dataloaders(args.dataset, args.batch_size)

    # Dataset normalization parameters
    if args.dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    base_model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    model = NormalizedModel(base_model, mean=mean, std=std).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        step_fn=METHODS[args.mode],
        epochs=args.epochs,
        device=device,
        save_name=f"best_{args.mode}_{args.dataset}.pth"
    )

if __name__ == "__main__":
    main()