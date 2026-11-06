"""Рантайм: загрузка чекпоинта, инференс, планирование, метрика «удивления».

Разделение на два контура — главная практическая идея:

  * быстрый (System 1): один forward энкодера + предиктора, ~единицы мс;
  * медленный (System 2): CEM-планирование, десятки-сотни мс.

Переключение по «удивлению» (surprise): если ошибка предсказания очередного
наблюдения выше порога, значит мир повёл себя не так, как модель ожидала —
пора думать, а не рефлекторно действовать. Это дешёвая и работающая
аппроксимация метакогниции.

Важная деталь: предиктор работает с контекстом из `ctx` латентов, поэтому
движок хранит историю. Без неё нельзя восстановить скорость объекта.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..planning.cem import CEMConfig, CEMPlanner


@dataclass
class RuntimeConfig:
    device: str = "auto"
    plan_horizon: int = 8
    plan_samples: int = 256
    plan_iters: int = 3
    surprise_threshold: float = 0.35
    action_clip: float = 1.0


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_ctx(model: torch.nn.Module) -> int:
    """Достаёт размер контекста из конфига модели (JEPA или HJEPA)."""
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return 1
    if hasattr(cfg, "ctx"):
        return int(cfg.ctx)
    base = getattr(cfg, "base", None)
    if base is not None and hasattr(base, "ctx"):
        return int(base.ctx)
    return 1


class OpenJEPAEngine:
    """Единая точка входа для инференса."""

    def __init__(self, model: torch.nn.Module, cfg: dict, runtime: RuntimeConfig | None = None):
        self.cfg = cfg
        self.rt = runtime or RuntimeConfig()
        self.device = _pick_device(self.rt.device)
        self.model = model.to(self.device).eval()
        self.ctx = model_ctx(self.model)
        self.planner = CEMPlanner(
            self.model,
            CEMConfig(
                horizon=self.rt.plan_horizon,
                n_samples=self.rt.plan_samples,
                iters=self.rt.plan_iters,
            ),
            action_dim=cfg["model"]["action_dim"],
        )
        self._hist: deque[torch.Tensor] = deque(maxlen=self.ctx)
        self._prev_ctx: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(cls, path: str | Path, runtime: RuntimeConfig | None = None):
        from ..train.pretrain import build_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model, _ = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        return cls(model, ckpt["config"], runtime)

    # ------------------------------------------------------------------ #
    def _to_tensor(self, obs) -> torch.Tensor:
        """np.uint8 (H,W,3) | np.float (H,W,3) | torch (C,H,W) -> (1,C,H,W)."""
        if isinstance(obs, torch.Tensor):
            x = obs.detach().float()
        else:
            arr = np.asarray(obs)
            if arr.dtype == np.uint8:
                x = torch.from_numpy(arr).float() / 255.0
            else:
                x = torch.from_numpy(arr.astype(np.float32))
            if x.ndim == 3 and x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if float(x.max()) > 1.5:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        return x.to(self.device)

    def reset(self) -> None:
        self._hist.clear()
        self._prev_ctx = None
        self._prev_action = None

    def _push(self, z: torch.Tensor) -> torch.Tensor:
        """Добавляет латент в историю и возвращает контекст (1,ctx,D)."""
        if not self._hist:
            for _ in range(self.ctx):
                self._hist.append(z)
        else:
            self._hist.append(z)
        return torch.stack(list(self._hist), dim=1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, obs) -> torch.Tensor:
        return self.model.encode(self._to_tensor(obs))

    @torch.no_grad()
    def surprise(self, obs) -> float:
        """Насколько текущее наблюдение неожиданно для модели.

        Если истории нет, удивление = 0.
        """
        z = self.encode(obs)
        if self._prev_ctx is None or self._prev_action is None:
            self._prev_ctx = self._push(z)
            return 0.0
        pred = self.model.predict_next(self._prev_ctx, self._prev_action.unsqueeze(0))
        s = float((pred - z).pow(2).mean().sqrt())
        self._prev_ctx = self._push(z)
        return s

    @torch.no_grad()
    def act(self, obs, goal, use_planner: bool = True) -> np.ndarray:
        """Возвращает действие (A,) в диапазоне [-1, 1].

        `use_planner=False` оставлен для совместимости: быстрый контур —
        это отдельная обученная политика (`GoalEncoder`), которой в этом
        чекпоинте нет, поэтому по умолчанию всегда используется CEM.
        """
        z = self.encode(obs)
        ctx = self._push(z)

        if isinstance(goal, torch.Tensor) and goal.ndim >= 2:
            zg = goal
        else:
            zg = self.encode(goal)
        if zg.ndim == 1:
            zg = zg.unsqueeze(0)

        actions, _ = self.planner.plan(ctx, zg, prev_action=self._prev_action)
        action = actions[:, 0].clamp(-self.rt.action_clip, self.rt.action_clip)

        self._prev_ctx = ctx
        self._prev_action = action.squeeze(0)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def observe(self, obs) -> None:
        """Обновить внутреннее состояние без выбора действия."""
        self._prev_ctx = self._push(self.encode(obs))


def make_goal_frame(env, pos: np.ndarray) -> np.ndarray:
    """Кадр-цель: мяч в нужной точке, скорость нулевая. Только для демо/тестов."""
    saved_pos, saved_vel = env.pos.copy(), env.vel.copy()
    env.pos = np.asarray(pos, dtype=np.float64).copy()
    env.vel = np.zeros(2)
    frame = env.render()
    env.pos, env.vel = saved_pos, saved_vel
    return frame
