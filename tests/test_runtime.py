"""Тесты рантайма: загрузка, состояние, surprise, act, HTTP-декодирование."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.models.jepa import JEPA, JEPAConfig  # noqa: E402
from openjepa.runtime.engine import OpenJEPAEngine, RuntimeConfig, make_goal_frame  # noqa: E402
from openjepa.data.synthetic import BallEnv, BallEnvConfig  # noqa: E402


def make_engine(ctx: int = 2) -> OpenJEPAEngine:
    cfg = JEPAConfig(
        img_size=32,
        patch_size=8,
        dim=64,
        depth=2,
        heads=2,
        proj_dim=16,
        proj_hidden=64,
        action_dim=2,
        ctx=ctx,
        pred_depth=2,
        pred_hidden=64,
        sigreg_slices=32,
    )
    model = JEPA(cfg)
    raw = {
        "model": {
            "img_size": 32,
            "patch_size": 8,
            "dim": 64,
            "depth": 2,
            "heads": 2,
            "proj_dim": 16,
            "proj_hidden": 64,
            "action_dim": 2,
            "ctx": ctx,
            "pred_depth": 2,
            "pred_hidden": 64,
            "sigreg_slices": 32,
            "hierarchical": False,
        }
    }
    return OpenJEPAEngine(
        model,
        raw,
        RuntimeConfig(device="cpu", plan_horizon=3, plan_samples=16, plan_iters=1),
    )


def test_engine_state_shapes():
    eng = make_engine(ctx=2)
    env = BallEnv(BallEnvConfig(size=32), seed=0)
    frame = env.reset()
    eng.set_goal(frame)
    a, s = eng.step(frame)
    assert a.shape == (2,)
    assert np.all(np.abs(a) <= 1.0 + 1e-6)
    assert len(eng._hist) == 2
    assert np.isfinite(s)


def test_step_pushes_frame_once():
    """step() не должен дублировать кадр в истории (иначе ломается скорость)."""
    eng = make_engine(ctx=2)
    env = BallEnv(BallEnvConfig(size=32), seed=4)
    frame = env.reset()
    eng.set_goal(frame)
    eng.step(frame)
    eng.step(frame)
    eng.step(frame)
    assert len(eng._hist) == 2  # deque(maxlen=2), но кадров добавлено ровно 3
    z = eng.encode(frame)
    assert torch.allclose(eng._hist[-1], z)


def test_surprise_is_zero_on_first_frame_then_positive():
    eng = make_engine()
    env = BallEnv(BallEnvConfig(size=32), seed=1)
    f0 = env.reset()
    assert eng.surprise(f0) == 0.0
    for _ in range(3):
        f = env.step(np.zeros(2))
        eng.observe(f)
    f = env.step(np.array([1.0, 0.0]))
    s = eng.surprise(f)
    assert s >= 0.0
    assert np.isfinite(s)


def test_reset_clears_state():
    eng = make_engine()
    env = BallEnv(BallEnvConfig(size=32), seed=2)
    f = env.reset()
    eng.set_goal(f)
    eng.step(f)
    assert len(eng._hist) > 0
    eng.reset()
    assert len(eng._hist) == 0
    assert eng._ctx is None


def test_goal_frame_does_not_disturb_env():
    env = BallEnv(BallEnvConfig(size=32), seed=3)
    env.reset()
    p0, v0 = env.pos.copy(), env.vel.copy()
    _ = make_goal_frame(env, np.array([10.0, 20.0]))
    assert np.allclose(env.pos, p0)
    assert np.allclose(env.vel, v0)


def test_to_tensor_accepts_uint8_and_float():
    eng = make_engine()
    u8 = np.zeros((32, 32, 3), dtype=np.uint8)
    x1 = eng._to_tensor(u8)
    x2 = eng._to_tensor(u8.astype(np.float32) / 255.0)
    assert x1.shape == (1, 3, 32, 32)
    assert torch.allclose(x1, x2, atol=1e-5)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
