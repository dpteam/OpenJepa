"""Тесты среды и датасета: физика, границы, формы, детерминизм."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.data.synthetic import (  # noqa: E402
    BallEnv,
    BallEnvConfig,
    TrajectoryDataset,
    collate,
    smooth_random_actions,
)


def test_render_shape_and_dtype():
    env = BallEnv(BallEnvConfig(size=48))
    frame = env.reset()
    assert frame.shape == (48, 48, 3)
    assert frame.dtype == np.uint8


def test_gravity_pulls_down():
    env = BallEnv(BallEnvConfig(gravity=0.2, drag=0.0, restitution=0.0))
    env.reset(pos=np.array([32.0, 10.0]), vel=np.zeros(2))
    y0 = env.pos[1]
    for _ in range(10):
        env.step(np.zeros(2))
    assert env.pos[1] > y0, "без гравитации мяч должен падать вниз (y растёт)"


def test_ball_stays_inside():
    env = BallEnv(BallEnvConfig(size=32, radius=4), seed=1)
    rng = np.random.default_rng(1)
    env.reset()
    for _ in range(300):
        env.step(rng.uniform(-1, 1, size=2))
        r, s = env.cfg.radius, env.cfg.size
        assert r - 1e-6 <= env.pos[0] <= s - r + 1e-6
        assert r - 1e-6 <= env.pos[1] <= s - r + 1e-6


def test_action_is_clipped():
    env = BallEnv(BallEnvConfig(accel_scale=1.0, gravity=0.0, drag=0.0))
    env.reset(pos=np.array([32.0, 32.0]), vel=np.zeros(2))
    env.step(np.array([100.0, 100.0]))
    assert np.linalg.norm(env.vel) <= env.cfg.max_speed + 1e-6


def test_dataset_deterministic():
    ds = TrajectoryDataset(n_traj=4, length=4, seed=42)
    a = ds[2]
    b = ds[2]
    assert torch.equal(a["frames"], b["frames"])
    assert torch.equal(a["actions"], b["actions"])
    assert a["frames"].shape == (5, 3, 64, 64)
    assert a["actions"].shape == (4, 2)


def test_collate_shapes():
    ds = TrajectoryDataset(n_traj=3, length=4, seed=0)
    batch = collate([ds[0], ds[1]])
    assert batch["frames"].shape == (2, 5, 3, 64, 64)
    assert batch["actions"].shape == (2, 4, 2)


def test_smooth_actions_are_bounded_and_smooth():
    rng = np.random.default_rng(0)
    a = smooth_random_actions(rng, 64, dim=2)
    assert a.shape == (64, 2)
    assert np.abs(a).max() <= 1.0 + 1e-6
    jerk = np.abs(np.diff(a, axis=0)).mean()
    assert jerk < 0.5, f"действия слишком рваные: {jerk:.3f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
