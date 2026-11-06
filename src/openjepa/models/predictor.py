"""Предиктор в латентном пространстве: ẑ_{t+1} = P(z_{t-k..t}, a_t).

Это сердце JEPA. В отличие от генеративной модели, предиктор НЕ восстанавливает
пиксели: он предсказывает эмбеддинг будущего кадра. Всё, что не поддаётся
предсказанию (шум, текстуры, случайные детали), автоматически выбрасывается
энкодером — именно это Лекун называет «абстрактным предсказанием».

Зачем нужен контекст из нескольких кадров (`ctx`):
    по одному кадру нельзя восстановить скорость объекта. Мяч в точке (30,30)
    может лететь вверх, вниз или стоять на месте. Если предиктор видит только
    z_t, задача предсказания z_{t+1} **неоднозначна**, и ошибка упирается в
    «неснижаемый» пол — сколько ни обучай. Контекст из 2–3 латентов снимает
    неоднозначность: разность соседних латентов играет роль производной.
    Это же решение используется в V-JEPA 2-AC и LeWM.

Условие по действию подаётся через FiLM (scale/shift), а не конкатенацией:
так предиктор остаётся residual-сетью, а действие модулирует признаки, а не
добавляет размерность входа. Это устойчивее при смене набора действий.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def context_windows(z: torch.Tensor, ctx: int) -> torch.Tensor:
    """Собирает окна прошлых латентов.

    Args:
        z: (B, T, D) — последовательность латентов.
        ctx: длина окна.

    Returns:
        (B, T, ctx, D), где окно `t` заканчивается на `z[:, t]` и содержит
        `z[:, t-ctx+1 : t+1]`. Слева последовательность дополняется копией
        `z[:, 0]`, поэтому окно определено и для `t < ctx-1`, и мы не теряем
        первые кадры.
    """
    if ctx < 1:
        raise ValueError("ctx должен быть >= 1")
    b, t, d = z.shape
    if ctx == 1:
        return z.unsqueeze(2)
    pad = z[:, :1].expand(b, ctx - 1, d)
    zp = torch.cat([pad, z], dim=1)  # (B, T+ctx-1, D)
    w = zp.unfold(1, ctx, 1)  # (B, T, D, ctx)
    return w.permute(0, 1, 3, 2)  # (B, T, ctx, D)


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.film = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + self.mlp(h)


class LatentPredictor(nn.Module):
    """Предиктор с условием на действие и контекстом из `ctx` латентов.

    Args:
        dim: размерность латентного пространства.
        action_dim: размерность вектора действия.
        hidden: ширина FiLM-блоков.
        depth: число FiLM-блоков.
        ctx: сколько последних латентов подаётся на вход.
        horizon: сколько шагов вперёд предсказывать за один вызов.
        action_hidden: ширина энкодера действия.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        hidden: int = 384,
        depth: int = 3,
        ctx: int = 1,
        horizon: int = 1,
        action_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.ctx = ctx
        self.horizon = horizon

        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_hidden, hidden),
        )
        # Для horizon>1 кодируем весь чанк действий одним вектором: модель
        # должна знать не только «куда», но и «сколько шагов».
        self.chunk_enc = nn.Sequential(
            nn.Linear(action_dim * horizon, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, hidden),
        )
        self.inp = nn.Linear(dim * ctx, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.out = nn.Linear(hidden, dim * horizon)
        # Нулевая инициализация выхода: на старте предиктор = identity,
        # обучение начинается с малых поправок. Дёшево и очень стабилизирует.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------ #
    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) или (B, ctx, D) -> (B, ctx*D)."""
        if z.ndim == 2:
            z = z.unsqueeze(1)
            if self.ctx > 1:
                z = z.expand(-1, self.ctx, -1)
        if z.ndim != 3:
            raise ValueError(f"ожидался (B,D) или (B,ctx,D), получено {tuple(z.shape)}")
        if z.shape[1] != self.ctx:
            raise ValueError(f"контекст {z.shape[1]} не совпадает с ctx={self.ctx}")
        return z.reshape(z.shape[0], -1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z: (B,D) или (B,ctx,D); action: (B,A) или (B,H,A) при horizon>1.

        Возвращает (B,D) при horizon==1, иначе (B,H,D).
        """
        zf = self._flatten(z)

        if self.horizon > 1:
            if action.ndim != 3:
                raise ValueError(
                    f"при horizon={self.horizon} ожидался action (B,H,A), "
                    f"получено {tuple(action.shape)}"
                )
            b, h, a = action.shape
            if h != self.horizon:
                raise ValueError(f"ожидался горизонт {self.horizon}, получено {h}")
            cond = self.chunk_enc(action.reshape(b, h * a))
            return self._core(zf, cond).reshape(b, h, self.dim)

        if action.ndim == 3:
            raise ValueError("action должен быть (B,A) при horizon=1")
        return self._core(zf, self.action_enc(action))

    def _core(self, zf: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.inp(zf)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(self.out_norm(h))


class InverseDynamics(nn.Module):
    """Предсказывает действие по паре соседних латентов: (z_t, z_{t+1}) → a_t.

    Зачем это нужно. Основная потеря JEPA — предсказание следующего латента —
    имеет ловушку: энкодер может выбросить **управляемые** факторы, если они
    не помогают предсказанию. В нашей среде гравитация делает вертикальное
    движение почти детерминированным, а горизонтальное зависит от действия.
    Энкодер выучивал только вертикаль: латентное расстояние между мячом
    слева и справа было 0.76 против 8.0 между верхом и низом. Планировщик
    физически не мог двигать объект вбок.

    Inverse-dynamics ломает эту симметрию: чтобы восстановить `a_t` из
    `(z_t, z_{t+1})`, латент обязан содержать всё, на что влияет действие.
    Это ровно тот приём, который H-JEPA применяет к реальным видео DROID
    [arXiv:2610.06805], а DINO-WM — к своим средам [arXiv:2411.04983].

    Стоимость — один маленький MLP. Польза — управляемость латента.
    """

    def __init__(self, dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim * 2, hidden), nn.GELU(approximate="tanh")]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))


class GoalEncoder(nn.Module):
    """Быстрая политика: (текущее состояние, цель) -> действие.

    Используется в runtime, когда нужно действие без CEM (System 1).
    В этом репозитории не обучается — оставлено как точка расширения.
    """

    def __init__(self, dim: int, action_dim: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(approximate="tanh"),
            *[
                layer
                for _ in range(depth)
                for layer in (nn.Linear(hidden, hidden), nn.GELU(approximate="tanh"))
            ],
            nn.Linear(hidden, action_dim),
        )

    def forward(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, goal], dim=-1))
