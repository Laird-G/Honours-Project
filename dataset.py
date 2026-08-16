import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(dataset="cifar10", batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Keeps data in [0.0, 1.0]
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    if dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset.lower() == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    return trainloader, testloader, num_classes