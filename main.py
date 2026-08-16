import argparse
import torch
import torch.nn as nn
from utils.data_loader import get_dataloaders
from models.wideresnet import WideResNet
from algorithms.train_clean import train_clean_model
from algorithms.train_pgd import train_pgd_model

torch.backends.cudnn.benchmark = False

class NormalizedModel(nn.Module):
    """
    Wraps model to apply dataset normalization on-the-fly inside forward().
    Ensures PGD attack clipping [0, 1] and clean test images pass through identical scaling.
    """
    def __init__(self, base_model, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)

def main():
    parser = argparse.ArgumentParser(description="Clean vs PGD Adversarial Training Baseline")
    parser.add_argument('--mode', type=str, default='pgd_at', choices=['clean', 'pgd_at'], help="Training mode")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(f" [1/4] Starting Initialization ({args.mode.upper()} Mode)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Active Compute Device: {device}", flush=True)

    print("\n [2/4] Loading Dataset...", flush=True)
    trainloader, testloader, num_classes = get_dataloaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n [3/4] Instantiating WideResNet (28-10) with On-the-Fly Normalization...", flush=True)
    base_model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    
    if args.dataset.lower() == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    model = NormalizedModel(base_model, mean=mean, std=std).to(device)

    print(f"\n [4/4] Executing {args.mode.upper()} Training Loop...", flush=True)
    if args.mode == 'clean':
        train_clean_model(model, trainloader, testloader, device, epochs=args.epochs)
    elif args.mode == 'pgd_at':
        train_pgd_model(model, trainloader, testloader, device, epochs=args.epochs)

if __name__ == "__main__":
    main()