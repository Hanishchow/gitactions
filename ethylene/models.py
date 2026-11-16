"""Models: constrained MLP, stoichiometric penalty, autoencoder, tiny transformer.

Inputs are standardised 6-vectors ordered
[rs_r0, c_comp, ldr_raw, ldr_blk, temp_c, rh]; c_comp is index 1.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConstrainedMLP(nn.Module):
    """6-256-128-64-32-1, ReLU hidden, linear output (45,057 params)."""

    def __init__(self) -> None:
        super().__init__()
        sizes = (6, 256, 128, 64, 32)
        layers: list[nn.Module] = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def stoich_penalty(f: Callable[[Tensor], Tensor], x: Tensor, c_idx: int = 1) -> Tensor:
    """mean(relu(-df/dx[:, c_idx])**2), differentiable w.r.t. f's parameters.

    Works on a detached clone of ``x`` so the caller's tensor never gains
    ``requires_grad``.
    """
    with torch.enable_grad():
        xg = x.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(f(xg).sum(), xg, create_graph=True)
    return F.relu(-grad[:, c_idx]).pow(2).mean()


def composite_loss(
    model: Callable[[Tensor], Tensor], x: Tensor, y: Tensor, lam_phys: float = 1.0
) -> Tensor:
    """MSE + lam_phys * stoich penalty (L2 is the optimizer's weight_decay)."""
    return F.mse_loss(model(x), y) + lam_phys * stoich_penalty(model, x)


class Autoencoder(nn.Module):
    """6-16-5-16-6 anomaly autoencoder; ``score`` is per-sample reconstruction MSE."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 5),
            nn.Linear(5, 16), nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def score(self, x: Tensor) -> Tensor:
        return (self(x) - x).pow(2).mean(dim=1)


class TinyTransformer(nn.Module):
    """Window encoder: (B, width, 6) -> (B,). d_model 32, 2 heads, 2 layers, ff 64."""

    def __init__(self, width: int = 16, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(6, d_model)
        self.pos = nn.Parameter(torch.randn(1, width, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=2, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, w: Tensor) -> Tensor:
        h = self.encoder(self.embed(w) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)
