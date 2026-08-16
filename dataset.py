import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset="cifar10", batch_size=256, num_workers=4, return_val=False, val_split=0.05):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    cls = torchvision.datasets.CIFAR10 if dataset.lower() == "cifar10" else torchvision.datasets.CIFAR100
    num_classes = 10 if dataset.lower() == "cifar10" else 100

    full_trainset = cls(root="./data", train=True, download=True, transform=transform_train)
    testset = cls(root="./data", train=False, download=True, transform=transform_test)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    if not return_val:
        trainloader = DataLoader(
            full_trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        return trainloader, testloader, num_classes

    # Stratified validation split for GPM Task 2 plateau scheduling
    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(
        full_trainset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    valloader = DataLoader(
        valset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    return trainloader, valloader, testloader, num_classes