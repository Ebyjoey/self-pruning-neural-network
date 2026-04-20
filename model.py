import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        # init near 0: sigmoid(0.1)=0.525 (open), gradient=0.249 (near maximum)
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 0.1))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        hard_gates = (gates > 0.5).float() + gates - gates.detach()
        return F.linear(x, self.weight * hard_gates, self.bias)

    def gate_values(self):
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def num_gates(self):
        return self.gate_scores.numel()

    def sparsity_loss(self):
        return torch.sigmoid(self.gate_scores).sum()


class PrunableNet(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):
        super().__init__()
        self.fc1 = PrunableLinear(input_dim, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def all_gates(self):
        return torch.cat([
            self.fc1.gate_values().view(-1),
            self.fc2.gate_values().view(-1),
        ])

    def total_gates(self):
        return self.fc1.num_gates() + self.fc2.num_gates()

    def sparsity_loss(self):
        # sum so per-gate gradient = lambda (not lambda/N)
        return self.fc1.sparsity_loss() + self.fc2.sparsity_loss()

    def gate_params(self):
        return [self.fc1.gate_scores, self.fc2.gate_scores]

    def non_gate_params(self):
        gate_ids = {id(p) for p in self.gate_params()}
        return [p for p in self.parameters() if id(p) not in gate_ids]
