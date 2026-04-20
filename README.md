# Self-Pruning Neural Network — CIFAR-10

Feedforward network that prunes its own weights via learnable sigmoid gates with stable sparsity–accuracy tradeoff.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

Trains four experiments (λ ∈ {1e-6, 5e-6, 1e-5, 5e-5}) for 30 epochs each.

## Outputs

| File | Description |
|------|-------------|
| `outputs/results.csv` | Lambda, test accuracy, sparsity % per run |
| `outputs/gate_distribution.png` | Gate histogram for best-accuracy model |

## Stability design

| Mechanism | Detail |
|-----------|--------|
| Warmup | Epochs 1–5 CE only; sparsity loss added from epoch 6 |
| Sparsity loss | `mean(sigmoid(gates))` — normalized, not sum |
| Gradient clipping | `clip_grad_norm_(..., max_norm=1.0)` |
| Gate init | `gate_scores = −1` → `sigmoid(−1) ≈ 0.27` |
| Hard gating | Straight-through estimator at threshold 0.1 |
