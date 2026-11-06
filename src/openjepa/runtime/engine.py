"""Рантайм: загрузка чекпоинта, инференс, планирование, метрика «удивления».

Разделение на два контура — главная практическая идея:

  * быстрый (System 1): один forward энкодера + предиктора, ~единицы мс;
  * медленный (System 2): CEM-планирование, десятки-сотни мс.

Переключение по «удивлению» (surprise): если ошибка предсказания очередного
наблюдения выше порога, значит мир повёл себя не так, как модель ожидала —
пора думать, а не рефлекторно действовать. Это дешёвая и работающая
аппроксимация метакогниции.

Две важные детали, которые легко испортить:

1. **История.** Предиктор работает с контекстом из `ctx` латентов. Движок
   хранит окно и сдвигает его на каждом новом кадре. Если случайно добавить
   один и тот же кадр дважды (например, вызвать `surprise()` и `act()` на
   одном наблюдении), «производная» исказится вдвое, и планировщик начнёт
   ошибаться. Поэтому основной метод — `step()`: он обрабатывает кадр ровно
   один раз.

2. **Цель.** Кадр-цель кодируется один раз через `set_goal()`. Иначе каждый
   шаг тратит целый forward энкодера на перекодирование одной и той же цели.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..planning.cem import CEMConfig, CEMPlanner, CostMode
from ..data.synthetic import normalize_position


@dataclass
class RuntimeConfig:
    device: str = "auto"
    plan_horizon: int = 8
    plan_samples: int = 256
    plan_iters: int = 3
    surprise_threshold: float = 0.35
    action_clip: float = 1.0
    cost_mode: str = "probe"   # "probe" (цель — позиция) или "latent"


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
    """Единая точка входа для инференса.

    Типичный цикл:

        eng = OpenJEPAEngine.from_checkpoint("runs/toy/best.pt")
        eng.reset()
        eng.set_goal(goal_frame)
        for frame in camera:
            action, surprise = eng.step(frame)
            robot.apply(action)
    """

    def __init__(self, model: torch.nn.Module, cfg: dict, runtime: RuntimeConfig | None = None):
        self.cfg = cfg
        self.rt = runtime or RuntimeConfig()
        self.device = _pick_device(self.rt.device)
        self.model = model.to(self.device).eval()
        self.ctx = model_ctx(self.model)
        probe = getattr(self.model, "probe", None)
        mode = CostMode(self.rt.cost_mode)
        if mode == CostMode.PROBE and probe is None:
            mode = CostMode.LATENT  # модель без пробы — откатываемся
        self.cost_mode = mode
        self.planner = CEMPlanner(
            self.model,
            CEMConfig(
                horizon=self.rt.plan_horizon,
                n_samples=self.rt.plan_samples,
                iters=self.rt.plan_iters,
                mode=mode,
            ),
            action_dim=cfg["model"]["action_dim"],
            probe=probe,
        )
        self._hist: deque[torch.Tensor] = deque(maxlen=self.ctx)
        self._ctx: torch.Tensor | None = None       # окно, заканчивающееся на t
        self._prev_action: torch.Tensor | None = None
        self._goal: torch.Tensor | None = None

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
        self._ctx = None
        self._prev_action = None

    def set_goal(self, goal) -> None:
        """Задаёт цель.

        В режиме `probe` цель — **физическая позиция** объекта, `(P,)` или
        кадр (тогда позиция извлекается пробой).
        В режиме `latent` — латент `(D,)` или кадр.

        Позиция — более честная постановка: «мяч в точке X», а не «мяч в
        точке X с нулевой скоростью». См. research/notes/2026-11-06-goal-definition.md.
        """
        if self.cost_mode == CostMode.PROBE:
            g = self._as_position(goal)
            self._goal = g.reshape(1, -1)
            return

        d = self.model.cfg.proj_dim
        is_latent = isinstance(goal, torch.Tensor) and goal.ndim in (1, 2) and goal.shape[-1] == d
        if is_latent:
            zg = goal.to(self.device)
        else:
            zg = self.encode(goal)
        self._goal = zg.reshape(1, -1)

    def _as_position(self, goal) -> torch.Tensor:
        """Приводит цель к нормированному вектору координат в [-1, 1].

        Принимает:
          * `(P,)` numpy/tensor — позиция (в пикселях или уже нормированная);
          * кадр `(H,W,3)`, `(C,H,W)`, `(B,C,H,W)` — кодируем и берём пробу.
        """
        p_dim = self.planner.probe_dim
        size = self.cfg.get("data", {}).get("env", {}).get("size", 64)

        if isinstance(goal, torch.Tensor):
            if goal.ndim in (1, 2) and goal.shape[-1] == p_dim:
                p = goal.reshape(-1).to(self.device, dtype=torch.float32)
                # если значения выходят за [-1, 1] — считаем, что это пиксели
                if float(p.abs().max()) > 1.5:
                    p = torch.as_tensor(
                        normalize_position(p.cpu().numpy(), size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                return p
            frame = goal
        else:
            arr = np.asarray(goal)
            if arr.ndim == 1 and arr.shape[0] == p_dim:
                return torch.as_tensor(
                    normalize_position(arr, size),
                    dtype=torch.float32,
                    device=self.device,
                )
            frame = goal

        z = self.encode(frame)
        return self.model.probe(z.float()).reshape(-1)

    def _push(self, z: torch.Tensor) -> torch.Tensor:
        """Добавляет латент в историю и возвращает окно (1,ctx,D)."""
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
    def position(self, obs, denormalize: bool = False) -> torch.Tensor:
        """Позиция объекта по наблюдению (через пробу).

        Возвращает **нормированные** координаты в `[-1, 1]` — так проба
        обучалась. `denormalize=True` переводит их в пиксели сцены.

        Доступна только если модель обучена с `probe_weight > 0`.
        """
        if getattr(self.model, "probe", None) is None:
            raise ValueError("у модели нет пробы: обучите с probe_weight > 0")
        z = self.encode(obs)
        p = self.model.probe(z.float()).squeeze(0)
        if denormalize:
            size = self.cfg.get("data", {}).get("env", {}).get("size", 64)
            p = (p + 1.0) * 0.5 * float(size)
        return p

    @torch.no_grad()
    def step(self, obs, goal=None) -> tuple[np.ndarray | None, float]:
        """Обрабатывает один кадр: считает surprise, планирует, выдаёт действие.

        Возвращает `(action, surprise)`. `action is None`, если цель не задана.

        Это единственный метод, который следует вызывать в цикле управления:
        он гарантирует, что кадр попадает в историю ровно один раз.
        """
        z = self.encode(obs)

        surprise = 0.0
        if self._ctx is not None and self._prev_action is not None:
            pred = self.model.predict_next(self._ctx, self._prev_action.unsqueeze(0))
            surprise = float((pred - z).pow(2).mean().sqrt())

        ctx = self._push(z)
        self._ctx = ctx

        if goal is not None:
            self.set_goal(goal)
        if self._goal is None:
            return None, surprise

        actions, _ = self.planner.plan(ctx, self._goal, prev_action=self._prev_action)
        action = actions[:, 0].clamp(-self.rt.action_clip, self.rt.action_clip)
        self._prev_action = action.squeeze(0)
        return action.squeeze(0).cpu().numpy(), surprise

    # ------------------------------------------------------------------ #
    # Удобные обёртки. ВНИМАНИЕ: каждая из них добавляет кадр в историю,
    # поэтому не вызывайте `act()` и `surprise()` на одном и том же кадре —
    # используйте `step()`.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def surprise(self, obs) -> float:
        return self.step(obs, goal=None)[1]

    @torch.no_grad()
    def act(self, obs, goal=None) -> np.ndarray:
        action, _ = self.step(obs, goal)
        if action is None:
            raise ValueError("цель не задана: вызовите set_goal() или передайте goal")
        return action

    @torch.no_grad()
    def observe(self, obs) -> None:
        self.step(obs, goal=None)


def make_goal_frame(env, pos: np.ndarray) -> np.ndarray:
    """Кадр-цель: мяч в нужной точке, скорость нулевая. Только для демо/тестов."""
    saved_pos, saved_vel = env.pos.copy(), env.vel.copy()
    env.pos = np.asarray(pos, dtype=np.float64).copy()
    env.vel = np.zeros(2)
    frame = env.render()
    env.pos, env.vel = saved_pos, saved_vel
    return frame
