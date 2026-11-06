"""Тесты планировщика на аналитической модели мира.

Используем «идеальный» предиктор z_{t+1} = z_t + a_t. Для него задача CEM
решается точно, поэтому можно проверить не только формы, но и качество.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.planning.cem import CEMConfig, CEMPlanner  # noqa: E402
from openjepa.planning.mpc import LatentMPC, MPCConfig  # noqa: E402


class LinearWorld:
    """z_{t+1} = z_t + a_t, a_t in [-1,1]^2."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        zs = [z0]
        z = z0
        for t in range(actions.shape[1]):
            z = z + actions[:, t]
            zs.append(z)
        return torch.stack(zs, dim=1)


def test_planner_reaches_goal():
    torch.manual_seed(0)
    world = LinearWorld(dim=2)
    planner = CEMPlanner(
        world,
        CEMConfig(horizon=4, n_samples=512, n_elite=64, iters=6, init_std=1.0),
        action_dim=2,
    )
    z0 = torch.zeros(1, 2)
    goal = torch.tensor([[1.5, -1.0]])
    actions, info = planner.plan(z0, goal)
    traj = world.rollout(z0, actions)
    err = (traj[:, -1] - goal).norm(dim=-1).item()
    assert err < 0.25, f"CEM не дошёл до цели: err={err:.3f}"


def test_actions_in_bounds():
    planner = CEMPlanner(LinearWorld(), CEMConfig(horizon=3, n_samples=64, iters=2), 2)
    actions, _ = planner.plan(torch.zeros(1, 2), torch.ones(1, 2))
    assert actions.min() >= -1.0 - 1e-6
    assert actions.max() <= 1.0 + 1e-6


def test_plan_shapes_batched():
    planner = CEMPlanner(LinearWorld(), CEMConfig(horizon=5, n_samples=32, iters=2), 2)
    z0 = torch.zeros(3, 2)
    goal = torch.ones(3, 2)
    actions, info = planner.plan(z0, goal)
    assert actions.shape == (3, 5, 2)
    assert info["cost"].shape == (3,)


def test_mpc_executes_and_replans():
    world = LinearWorld()
    mpc = LatentMPC(world, MPCConfig(cem=CEMConfig(horizon=4, n_samples=128, iters=3)), 2)
    z = torch.zeros(1, 2)
    goal = torch.tensor([[1.0, 1.0]])
    for _ in range(6):
        a = mpc.act(z, goal)
        z = world.rollout(z, a.unsqueeze(1))[:, -1]
    assert (z - goal).norm().item() < 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
