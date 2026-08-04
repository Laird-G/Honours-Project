import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(dataset_name="cifar10", batch_size=128, num_workers=2):
    print(" -> Configuring data transforms...", flush=True)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print(f" -> Accessing '{dataset_name}' dataset (downloading if not found)...", flush=True)
    if dataset_name.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10' or 'cifar100'.")

    print(f" -> Building DataLoaders with batch_size={batch_size} and workers={num_workers}...", flush=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True,                                       # Guarantees all batches are uniform size
        persistent_workers=True if num_workers > 0 else False # Prevents worker tear-down deadlocks
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    print(" -> Data loaders ready.", flush=True)
    return trainloader, testloader, num_classes