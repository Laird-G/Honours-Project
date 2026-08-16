import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(dataset="cifar10", batch_size=256, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    cls = torchvision.datasets.CIFAR10 if dataset.lower() == "cifar10" else torchvision.datasets.CIFAR100
    num_classes = 10 if dataset.lower() == "cifar10" else 100

    trainset = cls(root="./data", train=True, download=True, transform=transform_train)
    testset = cls(root="./data", train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    return trainloader, testloader, num_classes