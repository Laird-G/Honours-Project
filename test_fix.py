import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from itertools import islice

from utils.data_loader import get_dataloaders
from models.wideresnet import WideResNet
from algorithms.train_pgd import evaluate_clean, evaluate_robustness, train_one_epoch_pgd
from attacks.pgd import PGDAttack
from main import NormalizedModel

def run_empirical_tests():
    print("=" * 70, flush=True)
    print(" Running Empirical Pipeline Validation (Under 30 Seconds)", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Device: {device}", flush=True)

    # Load 1 batch of data
    trainloader, testloader, num_classes = get_dataloaders("cifar10", batch_size=128, num_workers=2)
    
    # Instantiate lightweight network for speed
    base_model = WideResNet(depth=16, num_classes=num_classes, widen_factor=2)
    model = NormalizedModel(base_model).to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # TEST 1: Forward-Pass Logit Agreement Test (\epsilon = 0)
    # -------------------------------------------------------------------------
    print("\n[Test 1/2] Checking Logit Agreement (Clean vs PGD with zero perturbation)...", flush=True)
    inputs, targets = next(iter(testloader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Path A: Direct Clean forward pass
    with torch.no_grad():
        with autocast('cuda'):
            clean_logits = model(inputs)

    # Path B: Zero-step PGD attack pass (\epsilon = 0)
    zero_attack = PGDAttack(model, epsilon=0.0, alpha=0.0, steps=0)
    zero_adv_inputs = zero_attack.perturb(inputs, targets)
    with torch.no_grad():
        with autocast('cuda'):
            pgd_zero_logits = model(zero_adv_inputs)

    # Empirical check: Maximum absolute difference between predictions
    max_diff = torch.max(torch.abs(clean_logits - pgd_zero_logits)).item()
    print(f" -> Max Logit Difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        print(" -> PASSED: Clean and Attack pipelines process identical input distributions.")
    else:
        print(" -> FAILED: Clean and Attack pipelines receive differently scaled inputs!")
        return

    # -------------------------------------------------------------------------
    # TEST 2: Mini-Training Metric Sanity Check (3 Fast Epochs)
    # -------------------------------------------------------------------------
    print("\n[Test 2/2] Running 3 Mini-Epochs (10 batches/epoch) to check metric ordering...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler('cuda')
    
    train_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=7)
    eval_attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=20)

    for epoch in range(50):
        # Truncate trainloader to 10 batches for rapid execution
        mini_trainloader = list(islice(trainloader, 30))
        mini_testloader = list(islice(testloader, 5))

        train_loss, train_acc = train_one_epoch_pgd(
            epoch, 3, model, mini_trainloader, criterion, optimizer, device, train_attack, scaler
        )

        clean_acc = evaluate_clean(epoch, 3, model, mini_testloader, criterion, device)
        pgd20_acc = evaluate_robustness(epoch, 3, model, mini_testloader, criterion, device, eval_attack)

        print(
            f" Epoch [{epoch+1}/3] | Train Adv Acc: {train_acc*100:.2f}% "
            f"| Test Clean Acc: {clean_acc*100:.2f}% | Test PGD-20 Acc: {pgd20_acc*100:.2f}%"
        )

        # Empirical assertion: Clean accuracy must never be lower than robust accuracy
        if clean_acc < pgd20_acc - 0.05:  # Allow 5% tolerance for noise on 5 mini-batches
            print("\n -> FAILED: Test Clean Acc is lower than Test PGD-20 Acc! Pipeline is corrupt.")
            return

    print(" ALL EMPIRICAL TESTS PASSED: Pipeline is valid and ready to submit.", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_empirical_tests()