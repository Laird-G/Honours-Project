import argparse
import torch
from utils.data_loader import get_dataloaders
from models.wideresnet import WideResNet
from algorithms.train_clean import train_clean_model

def main():
    parser = argparse.ArgumentParser(description="Clean Training Baseline")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader, num_classes = get_dataloaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size
    )

    model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10).to(device)

    train_clean_model(model, trainloader, testloader, device, epochs=args.epochs)

if __name__ == "__main__":
    main()