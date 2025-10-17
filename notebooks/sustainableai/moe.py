import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

# ----- shared encoder -> embedding -----
class Encoder(nn.Module):
    def __init__(self, in_shape=(3, 28, 28), dim=128):
        super().__init__()
        c, h, w = in_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),  # 7x7
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, dim)
    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.proj(x)  # [B, D]

# ----- expert is a tiny MLP head -----
class Expert(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )
    def forward(self, x):
        return self.mlp(x)

# ----- Mixture-of-Experts with top-1 routing -----
class MoE(nn.Module):
    def __init__(self, in_shape=(3,28,28), num_classes=4, dim=128, num_experts=4, router_temp=1.0):
        super().__init__()
        self.enc = Encoder(in_shape, dim)
        self.router = nn.Linear(dim, num_experts)      # produces gating logits
        self.experts = nn.ModuleList([Expert(dim, num_classes) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.router_temp = router_temp

    def forward(self, x):
        z = self.enc(x)                                # [B, D]
        gate_logits = self.router(z) / self.router_temp
        gate_probs = F.softmax(gate_logits, dim=-1)    # [B, E]

        # top-1 dispatch
        top1 = gate_probs.argmax(dim=-1)               # [B]
        B = z.size(0)
        E = self.num_experts
        logits = z.new_zeros(B, self.experts[0].mlp[-1].out_features)

        # compute expert usage stats and aux balance loss (Switch-like)
        with torch.no_grad():
            counts = torch.bincount(top1, minlength=E).float() / B
        importance = gate_probs.mean(dim=0)            # how much probability mass each expert gets
        aux_loss = E * torch.sum(counts * importance)  # encourages equal usage and mass

        # dispatch by grouping indices per expert
        for e, expert in enumerate(self.experts):
            idx = (top1 == e).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            out_e = expert(z.index_select(0, idx))     # [n_e, C]
            logits.index_copy_(0, idx, out_e)

        return logits, aux_loss, counts.detach(), importance.detach()

# ----- minimal data plumbing -----
def collate_float_long(batch):
    x, y = default_collate(batch)
    x = x.float()
    if x.max() > 1.0:  # in case images are 0..255
        x = x / 255.0
    if isinstance(y, torch.Tensor) and y.ndim == 2:
        y = y.argmax(dim=1)   # collapse one-hot to indices
    return x, y.long()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, correct, aux_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, aux, _, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
        aux_sum += aux.item()
    return correct / tot, aux_sum / max(1, len(loader))
