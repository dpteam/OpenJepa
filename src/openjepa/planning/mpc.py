"""Receding-horizon control (MPC) поверх CEM-планировщика."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cem import CEMConfig, CEMPlanner


@dataclass
class MPCConfig:
    plan_every: int = 1
    warm_start: bool = True
    cem: CEMConfig = field(default_factory=CEMConfig)


class LatentMPC:
    """Планирует, исполняет первый шаг, перепланирует.

    Это и есть «System 2» из статьи Лекуна: медленный, но осмысленный контур
    управления. Быстрый контур (политика) добавляется отдельно и включается,
    когда ошибка предсказания мала.
    """

    def __init__(self, model, cfg: MPCConfig | None = None, action_dim: int = 2):
        self.model = model
        self.cfg = cfg or MPCConfig()
        self.planner = CEMPlanner(model, self.cfg.cem, action_dim=action_dim)
        self._prev: torch.Tensor | None = None
        self._steps_since_plan = 0
        self._plan: torch.Tensor | None = None
        self._plan_idx = 0

    def reset(self) -> None:
        self._prev = None
        self._plan = None
        self._plan_idx = 0
        self._steps_since_plan = 0

    @torch.no_grad()
    def act(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """z: (D,) или (B,D); goal: (D,). Возвращает действие (A,) или (B,A)."""
        squeeze = z.ndim == 1
        zz = z.unsqueeze(0) if squeeze else z

        need_plan = (
            self._plan is None
            or self._plan_idx >= self._plan.shape[1]
            or self._steps_since_plan >= self.cfg.plan_every
        )
        if need_plan:
            prev = self._prev if self.cfg.warm_start else None
            self._plan, _ = self.planner.plan(zz, goal, prev_action=prev)
            self._plan_idx = 0
            self._steps_since_plan = 0

        action = self._plan[:, self._plan_idx]
        self._plan_idx += 1
        self._steps_since_plan += 1
        self._prev = action
        return action.squeeze(0) if squeeze else action
