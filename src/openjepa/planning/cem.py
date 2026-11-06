"""Планирование в латентном пространстве: Cross-Entropy Method (CEM).

Почему CEM, а не градиенты: предиктор — нейросеть, но среда за ним не
дифференцируема по действиям в общем случае (контакты, отскоки, переключения).
CEM — это population-based оптимизация: сэмплируем N последовательностей
действий, «воображаем» их в латентном пространстве, оставляем лучшие и
подтягиваем распределение к ним. Ровно так работает MPC в V-JEPA 2-AC и
LeWorldModel.

Ключевое преимущество JEPA: стоимость оценивается по латентной траектории,
без декодирования в пиксели. Отсюда 48-кратное ускорение LeWM против
world-моделей на фундаментных моделях.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CEMConfig:
    horizon: int = 8
    n_samples: int = 256
    n_elite: int = 32
    iters: int = 3
    init_std: float = 0.6
    min_std: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    action_penalty: float = 0.02
    smooth_penalty: float = 0.05
    discount: float = 0.95


class CEMPlanner:
    """Оптимизирует последовательность действий по латентной цели."""

    def __init__(self, model, cfg: CEMConfig | None = None, action_dim: int = 2):
        self.model = model
        self.cfg = cfg or CEMConfig()
        self.action_dim = action_dim

    @torch.no_grad()
    def plan(
        self,
        z0: torch.Tensor,
        goal: torch.Tensor,
        prev_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """z0: (B,D) | (D,) | (B,ctx,D); goal: (D,) или (B,D) -> (B,H,A), info.

        Возвращает всю лучшую последовательность; вызывающий код обычно
        исполняет только первый шаг (receding horizon).
        """
        cfg = self.cfg
        squeeze = z0.ndim == 1
        if squeeze:
            z0 = z0.unsqueeze(0)
        if z0.ndim == 3:
            b, k, d = z0.shape
        elif z0.ndim == 2:
            b, d = z0.shape
        else:
            raise ValueError(f"z0 должен быть (B,D) или (B,ctx,D), получено {z0.shape}")

        goal = torch.as_tensor(goal, device=z0.device, dtype=z0.dtype)
        if goal.ndim == 1:
            goal = goal.reshape(1, -1)
        if goal.shape[-1] != d:
            raise ValueError(f"размерность цели {goal.shape[-1]} != {d}")
        if goal.shape[0] == 1:
            goal = goal.expand(b, d)
        elif goal.shape[0] != b:
            raise ValueError(f"батч цели {goal.shape[0]} не совпадает с {b}")

        mean = torch.zeros(b, cfg.horizon, self.action_dim, device=z0.device)
        std = torch.full_like(mean, cfg.init_std)
        if prev_action is not None:
            pa = prev_action.reshape(b, 1, self.action_dim)
            mean = mean + 0.5 * pa  # тёплый старт: не дёргаться без нужды

        best_cost = torch.full((b,), float("inf"), device=z0.device)
        best_actions = mean.clone()
        n_elite = max(1, min(cfg.n_elite, cfg.n_samples))

        for _ in range(cfg.iters):
            noise = torch.randn(
                b, cfg.n_samples, cfg.horizon, self.action_dim, device=z0.device
            )
            samples = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                cfg.action_low, cfg.action_high
            )
            cost = self._cost(z0, goal, samples)  # (B,S)

            elite_idx = cost.topk(n_elite, dim=1, largest=False).indices  # (B,E)
            elite = torch.gather(
                samples, 1, elite_idx[..., None, None].expand(-1, -1, cfg.horizon, self.action_dim)
            )
            mean = elite.mean(dim=1)
            std = elite.std(dim=1).clamp_min(cfg.min_std)

            step_best, step_idx = cost.min(dim=1)
            improved = step_best < best_cost
            best_cost = torch.where(improved, step_best, best_cost)
            best_actions = torch.where(
                improved[:, None, None], samples[torch.arange(b), step_idx], best_actions
            )

        info = {"cost": best_cost, "goal_latent_norm": goal.norm(dim=-1)}
        return (best_actions.squeeze(0) if squeeze else best_actions), info

    def _cost(self, z0: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: (B,S,H,A) -> (B,S). z0: (B,D) или (B,ctx,D)."""
        cfg = self.cfg
        b, s, h, a = actions.shape
        flat = actions.reshape(b * s, h, a)

        if z0.ndim == 3:
            _, k, d = z0.shape
            z_start = z0.unsqueeze(1).expand(b, s, k, d).reshape(b * s, k, d)
        else:
            z_start = z0.unsqueeze(1).expand(b, s, z0.shape[-1]).reshape(b * s, -1)

        traj = self.model.rollout(z_start, flat)  # (B*S, H+1, D)

        goal_rep = goal.unsqueeze(1).expand(b, s, goal.shape[-1]).reshape(b * s, -1)
        dist = (traj - goal_rep.unsqueeze(1)).pow(2).sum(-1)  # (B*S, H+1)

        weights = torch.tensor(
            [cfg.discount ** t for t in range(h + 1)], device=traj.device, dtype=traj.dtype
        )
        weights = weights / weights.sum()
        cost = (dist * weights).sum(-1)

        # Регуляризация: маленькие действия и плавность.
        cost = cost + cfg.action_penalty * flat.pow(2).mean(dim=(1, 2))
        if h > 1:
            smooth = (flat[:, 1:] - flat[:, :-1]).pow(2).mean(dim=(1, 2))
            cost = cost + cfg.smooth_penalty * smooth
        return cost.reshape(b, s)
