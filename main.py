import argparse
import torch
from utils.data_loader import get_dataloaders
from models.wideresnet import WideResNet
from algorithms.train_clean import train_clean_model
from algorithms.train_pgd import train_pgd_model

torch.backends.cudnn.benchmark = False

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

    print("\n [3/4] Instantiating WideResNet (28-10)...", flush=True)
    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10).to(device)

    print(f"\n [4/4] Executing {args.mode.upper()} Training Loop...", flush=True)
    if args.mode == 'clean':
        train_clean_model(model, trainloader, testloader, device, epochs=args.epochs)
    elif args.mode == 'pgd_at':
        train_pgd_model(model, trainloader, testloader, device, epochs=args.epochs)

if __name__ == "__main__":
    main()