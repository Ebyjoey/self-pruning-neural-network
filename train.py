import os
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PrunableNet

SEED        = 42
BATCH_SIZE  = 256
EPOCHS      = 30
WARMUP      = 5
LR          = 1e-3
GATE_LR     = 1e-2
LAMBDAS     = [1e-9, 1e-8, 5e-8, 2e-7]
DATA_DIR    = "./data"
OUT_DIR     = "./outputs"
GATE_THRESH = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    return (torch.utils.data.DataLoader(train_set, shuffle=True,  **kw),
            torch.utils.data.DataLoader(test_set,  shuffle=False, **kw))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100.0


def compute_sparsity(model):
    g = model.all_gates().cpu().numpy()
    return float((g < GATE_THRESH).mean()) * 100.0


def train_run(lam, train_loader, test_loader, device):
    set_seed(SEED)
    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam([
        {"params": model.non_gate_params(), "lr": LR},
        {"params": model.gate_params(),     "lr": GATE_LR},
    ])
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = ce(model(images), labels)
            if epoch > WARMUP:
                loss = loss + lam * model.sparsity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            acc = evaluate(model, test_loader, device)
            sp  = compute_sparsity(model)
            g_mean = model.all_gates().mean().item()
            print(f"  λ={lam:.0e}  epoch={epoch:02d}  acc={acc:.2f}%  "
                  f"sparsity={sp:.1f}%  gate_mean={g_mean:.3f}")

    acc = evaluate(model, test_loader, device)
    sp  = compute_sparsity(model)
    return model, acc, sp


def plot_gates(model, lam):
    gates = model.all_gates().cpu().numpy()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gates, bins=80, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.axvline(GATE_THRESH, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"threshold = {GATE_THRESH}")
    sp = float((gates < GATE_THRESH).mean()) * 100
    ax.text(0.97, 0.95, f"Sparsity: {sp:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#C44E52")
    ax.set_xlabel("Gate value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate distribution  |  λ = {lam:.0e}", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gate_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gate distribution saved → {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")
    train_loader, test_loader = get_loaders()

    results    = []
    best_model = None
    best_lam   = None
    best_acc   = -1.0

    for lam in LAMBDAS:
        print(f"─── λ = {lam:.0e} ───")
        model, acc, sp = train_run(lam, train_loader, test_loader, device)
        results.append({"lambda": lam, "accuracy": round(acc, 2), "sparsity": round(sp, 1)})
        print(f"  → final acc={acc:.2f}%  sparsity={sp:.1f}%\n")
        if acc > best_acc:
            best_acc, best_model, best_lam = acc, model, lam

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lambda", "accuracy", "sparsity"])
        w.writeheader()
        w.writerows(results)
    print(f"Results saved → {csv_path}")

    plot_gates(best_model, best_lam)

    print("\n=== Summary ===")
    print(f"{'Lambda':<12} {'Accuracy':>12} {'Sparsity %':>12}")
    print("─" * 38)
    for r in results:
        print(f"{r['lambda']:<12.0e} {r['accuracy']:>11.2f}% {r['sparsity']:>11.1f}%")


if __name__ == "__main__":
    main()
