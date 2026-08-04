import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter, calculate_accuracy
from torch.amp import autocast, GradScaler

def train_one_epoch(epoch, epochs, model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Train", 
        leave=False, 
        dynamic_ncols=True
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

        # Real-time progress updates inside terminal bar
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg * 100:.2f}%")

    return losses.avg, accs.avg

def evaluate(epoch, epochs, model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch [{epoch+1:03d}/{epochs:03d}] Eval ", 
        leave=False, 
        dynamic_ncols=True
    )

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))

            pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accs.avg * 100:.2f}%")

    return losses.avg, accs.avg

def train_clean_model(model, trainloader, testloader, device, epochs=200):
    print(" -> Setting up CrossEntropyLoss, SGD Optimizer, and Cosine Scheduler...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = GradScaler('cuda')
    best_acc = 0.0

    print("-" * 60, flush=True)
    print(" Training Started", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(epoch, epochs, model, trainloader, criterion, optimizer, device, scaler)
        test_loss, test_acc = evaluate(epoch, epochs, model, testloader, criterion, device)
        
        scheduler.step()

        saved_text = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_clean_model.pth")
            saved_text = " -> Saved Best Checkpoint!"

        print(
            f"Epoch [{epoch+1:03d}/{epochs:03d}] "
            f"| Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% "
            f"| Test Loss: {test_loss:.4f} - Acc: {test_acc*100:.2f}%"
            f"{saved_text}",
            flush=True
        )
    
    print("-" * 60, flush=True)
    print(f"Training complete. Best Standard Accuracy: {best_acc*100:.2f}%", flush=True)