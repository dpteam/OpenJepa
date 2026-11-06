"""Тесты моделей: формы, конечность потерь, rollout, иерархия."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.models.hierarchy import HJEPA, HierarchyConfig  # noqa: E402
from openjepa.models.jepa import JEPA, JEPAConfig  # noqa: E402


def small_cfg(**kw) -> JEPAConfig:
    base = dict(
        img_size=32,
        patch_size=8,
        dim=64,
        depth=2,
        heads=2,
        proj_dim=16,
        proj_hidden=64,
        action_dim=2,
        ctx=2,
        pred_depth=2,
        pred_hidden=64,
        sigreg_slices=32,
    )
    base.update(kw)
    return JEPAConfig(**base)


def test_encoder_shapes():
    m = JEPA(small_cfg())
    img = torch.randn(3, 3, 32, 32)
    vid = torch.randn(3, 5, 3, 32, 32)
    assert m.encoder(img).shape == (3, 64)
    assert m.encoder(vid).shape == (3, 5, 64)
    assert m.encode(img).shape == (3, 16)
    assert m.encode_sequence(vid).shape == (3, 5, 16)


def test_loss_is_finite_and_differentiable():
    m = JEPA(small_cfg())
    obs = torch.randn(2, 6, 3, 32, 32)
    act = torch.randn(2, 5, 2).clamp(-1, 1)
    out = m(obs, act)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rollout_shapes():
    m = JEPA(small_cfg())
    z0 = torch.randn(2, 16)
    acts = torch.randn(2, 4, 2).clamp(-1, 1)
    traj = m.rollout(z0, acts)
    assert traj.shape == (2, 5, 16)
    assert torch.allclose(traj[:, 0], z0)


def test_multistep_horizon():
    m = JEPA(small_cfg(horizon=3))
    obs = torch.randn(2, 6, 3, 32, 32)
    act = torch.randn(2, 5, 2).clamp(-1, 1)
    out = m(obs, act)
    assert torch.isfinite(out["loss"])
    z = torch.randn(2, 16)
    pred = m.predict_next(z, torch.randn(2, 3, 2))
    assert pred.shape == (2, 3, 16)


def test_context_windows():
    """Окно t должно заканчиваться на z_t и содержать ctx последних латентов."""
    from openjepa.models.predictor import context_windows

    z = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    w = context_windows(z, 2)
    assert w.shape == (2, 5, 2, 3)
    assert torch.allclose(w[:, 0], torch.stack([z[:, 0], z[:, 0]], dim=1))
    assert torch.allclose(w[:, 3, 0], z[:, 2])
    assert torch.allclose(w[:, 3, 1], z[:, 3])


def test_rollout_keeps_context():
    """После rollout история должна сдвигаться, а не сбрасываться."""
    m = JEPA(small_cfg())
    z = torch.randn(2, 2, 16)          # контекст из 2 латентов
    acts = torch.randn(2, 3, 2).clamp(-1, 1)
    traj = m.rollout(z, acts)
    assert traj.shape == (2, 4, 16)
    assert torch.allclose(traj[:, 0], z[:, -1])


def test_hierarchy_forward():
    cfg = HierarchyConfig(
        levels=3, strides=[1, 2, 4], base=small_cfg()
    )
    m = HJEPA(cfg)
    obs = torch.randn(2, 6, 3, 32, 32)
    act = torch.randn(2, 5, 2).clamp(-1, 1)
    out = m(obs, act)
    assert torch.isfinite(out["loss"])
    for i in range(3):
        assert f"pred_loss_l{i}" in out


def test_no_collapse_on_untrained_but_sigreg_present():
    """SIGReg должен быть положительным уже на случайных весах."""
    m = JEPA(small_cfg())
    obs = torch.randn(4, 4, 3, 32, 32)
    act = torch.randn(4, 3, 2).clamp(-1, 1)
    out = m(obs, act)
    assert float(out["sigreg_loss"]) > 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
