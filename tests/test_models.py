import pytest
import torch

from ethylene.models import (
    Autoencoder,
    ConstrainedMLP,
    TinyTransformer,
    composite_loss,
    stoich_penalty,
)


def _n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_mlp_param_count():
    assert _n_params(ConstrainedMLP()) == 45057


def test_mlp_output_shape():
    assert ConstrainedMLP()(torch.randn(5, 6)).shape == (5,)


def test_penalty_zero_for_increasing():
    x = torch.randn(32, 6)
    assert stoich_penalty(lambda t: t[:, 1], x).item() == 0.0


def test_penalty_one_for_decreasing():
    x = torch.randn(32, 6)
    assert stoich_penalty(lambda t: -t[:, 1], x).item() == pytest.approx(1.0)


def test_penalty_leaves_caller_input_untouched():
    x = torch.randn(8, 6)
    stoich_penalty(lambda t: -t[:, 1], x)
    assert not x.requires_grad
    assert x.grad is None


def test_penalty_is_differentiable():
    torch.manual_seed(0)
    model = ConstrainedMLP()
    x, y = torch.randn(16, 6), torch.randn(16)
    composite_loss(model, x, y, lam_phys=1.0).backward()
    assert all(p.grad is not None for p in model.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())


def test_composite_loss_lam_zero_is_plain_mse():
    torch.manual_seed(0)
    model = ConstrainedMLP()
    x, y = torch.randn(16, 6), torch.randn(16)
    plain = torch.nn.functional.mse_loss(model(x), y)
    assert composite_loss(model, x, y, lam_phys=0.0).item() == pytest.approx(plain.item())


def test_autoencoder_score_shape_and_learns():
    torch.manual_seed(0)
    ae = Autoencoder()
    x = torch.randn(64, 6)
    assert ae(x).shape == (64, 6)
    assert ae.score(x).shape == (64,)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    before = torch.nn.functional.mse_loss(ae(x), x).item()
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(ae(x), x)
        loss.backward()
        opt.step()
    assert loss.item() < before


def test_transformer_shape_and_size():
    model = TinyTransformer()
    assert model(torch.randn(3, 16, 6)).shape == (3,)
    assert _n_params(model) < 50_000
