"""Тесты SIGReg: форма, устойчивость, чувствительность к коллапсу."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.losses.sigreg import SIGReg  # noqa: E402


def test_shapes_and_finiteness():
    reg = SIGReg(num_slices=32)
    for shape in [(16, 8), (4, 5, 8), (2, 3, 4, 8)]:
        x = torch.randn(*shape)
        v = reg(x)
        assert v.shape == (), f"ожидался скаляр, получено {v.shape}"
        assert torch.isfinite(v), "статистика должна быть конечной"


def test_gradient_flows():
    reg = SIGReg(num_slices=32)
    x = torch.randn(32, 16, requires_grad=True)
    reg(x).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "градиент не должен быть нулевым"


def test_collapse_is_penalized():
    """Схлопнутое представление должно давать заметно большую статистику."""
    torch.manual_seed(0)
    reg = SIGReg(num_slices=256)

    good = torch.randn(1024, 32)
    collapsed = torch.zeros(1024, 32) + 0.001 * torch.randn(1024, 32)

    v_good = float(reg(good))
    v_bad = float(reg(collapsed))
    assert v_bad > v_good * 5, f"коллапс не штрафуется: good={v_good:.3f} bad={v_bad:.3f}"


def test_anisotropy_is_penalized():
    """Вытянутое облако (одна ось с большой дисперсией) хуже изотропного."""
    torch.manual_seed(0)
    reg = SIGReg(num_slices=256)
    iso = torch.randn(2048, 16)
    aniso = iso.clone()
    aniso[:, 0] *= 6.0
    assert float(reg(aniso)) > float(reg(iso))


def test_bounded_for_gaussian():
    """Для истинно гауссовых данных статистика ограничена и не растёт с N.

    Это и есть свойство «bounded loss» из статьи: в отличие от моментов,
    статистика Эппса–Пули не взрывается при увеличении выборки.
    """
    torch.manual_seed(0)
    reg = SIGReg(num_slices=256)
    vals = [float(reg(torch.randn(n, 32))) for n in (128, 512, 2048)]
    assert all(0.0 < v < 20.0 for v in vals), vals
    assert max(vals) < 4 * min(vals) + 5.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
