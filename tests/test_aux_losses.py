"""Тесты вспомогательных потерь: метрика и inverse-dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.losses.auxiliary import inverse_dynamics_loss, metric_loss  # noqa: E402
from openjepa.models.predictor import InverseDynamics  # noqa: E402


def test_metric_loss_zero_when_linear():
    """Если латент — изометричное масштабирование позиции, потеря ~0.

    Важно: именно **изометричное**. Анизотропное растяжение (разный масштаб
    по осям) даёт ненулевую потерю, потому что отношение ‖Δz‖/‖Δp‖ тогда
    зависит от направления. Для планировщика это плохо: одинаковый сдвиг
    по X и по Y должен «стоить» одинаково.
    """
    torch.manual_seed(0)
    pos = torch.rand(200, 2) * 10
    z = pos * 3.0
    loss = float(metric_loss(z, pos, n_pairs=128))
    assert loss < 0.02, f"изометрия должна давать ~0, получено {loss:.4f}"


def test_metric_loss_penalizes_anisotropy():
    """Растяжение по одной оси должно штрафоваться."""
    torch.manual_seed(0)
    pos = torch.rand(200, 2) * 10
    iso = pos * 3.0
    aniso = pos * torch.tensor([1.0, 4.0])
    assert float(metric_loss(aniso, pos, n_pairs=256)) > float(
        metric_loss(iso, pos, n_pairs=256)
    )


def test_metric_loss_positive_when_scrambled():
    """Если латент не связан с позицией, потеря должна быть большой."""
    torch.manual_seed(0)
    pos = torch.rand(200, 2) * 10
    z = torch.randn(200, 16)
    loss = float(metric_loss(z, pos, n_pairs=128))
    assert loss > 0.3, f"несвязанный латент должен давать большую потерю, получено {loss:.4f}"


def test_metric_loss_is_dimensionless():
    """Масштаб латента не должен влиять на значение потери."""
    torch.manual_seed(0)
    pos = torch.rand(200, 2) * 10
    z = torch.randn(200, 16) * 0.01
    l1 = float(metric_loss(z, pos, n_pairs=128))
    l2 = float(metric_loss(z * 1000, pos, n_pairs=128))
    assert abs(l1 - l2) < 0.05, f"потеря зависит от масштаба: {l1:.3f} vs {l2:.3f}"


def test_metric_loss_gradient_flows():
    torch.manual_seed(0)
    pos = torch.rand(64, 2) * 10
    z = torch.randn(64, 8, requires_grad=True)
    metric_loss(z, pos, n_pairs=32).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_metric_loss_handles_none_and_small_batches():
    z = torch.randn(4, 8)
    assert float(metric_loss(z, None)) == 0.0
    assert float(metric_loss(z[:1], torch.rand(1, 2))) == 0.0


def test_inverse_dynamics_loss_shapes():
    head = InverseDynamics(dim=8, action_dim=2, hidden=16)
    z = torch.randn(3, 6, 8)
    a = torch.randn(3, 5, 2)
    loss = inverse_dynamics_loss(head, z, a)
    assert loss.shape == ()
    assert torch.isfinite(loss)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
