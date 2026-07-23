import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter, calculate_accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

    return losses.avg, accs.avg

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))

    return losses.avg, accs.avg

def train_clean_model(model, trainloader, testloader, device, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}  | Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_clean_model.pth")
    
    print(f"Training complete. Best Standard Accuracy: {best_acc:.4f}")