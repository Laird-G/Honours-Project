import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(dataset="cifar10", batch_size=256, num_workers=4, return_val=False, val_split=0.05):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # No Normalize here on purpose: tensors stay in [0, 1] so PGD operates in
    # true pixel space. NormalizedModel applies mean/std inside forward().
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

    # 5% validation split (Sharmin 2022, Sec. 4.4.1), drawn deterministically.
    # Note this is a uniform random split, not a stratified one.
    #
    # The val subset is taken from a second, *un-augmented* view of the same
    # training data. random_split over the augmented set would have meant:
    #   (a) GPM bases extracted from randomly cropped/flipped images rather
    #       than the clean X_clean / A_clean that Sec. 4.3.2 calls for -- and
    #       non-deterministically, which is fatal for an l_th sweep; and
    #   (b) a noisy plateau metric driving ReduceLROnPlateau and best-model
    #       selection.
    full_evalset = cls(root="./data", train=True, download=True, transform=transform_test)

    val_size = int(len(full_trainset) * val_split)
    perm = torch.randperm(
        len(full_trainset), generator=torch.Generator().manual_seed(42)
    ).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    trainset = Subset(full_trainset, train_idx)   # augmented
    valset = Subset(full_evalset, val_idx)        # clean

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
