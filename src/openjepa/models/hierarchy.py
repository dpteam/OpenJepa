"""Иерархический JEPA (H-JEPA, arXiv:2610.06805).

Идея: один предиктор не может одновременно быть точным на 100 мс и осмысленным
на 10 с. Поэтому строится пирамида уровней:

    уровень 2 (медленный, абстрактный): z_{t+10s} из z_t
    уровень 1 (средний):               z_{t+1s}  из z_t и подцели сверху
    уровень 0 (быстрый, точный):       z_{t+100мс} из z_t и действия

Планирование идёт сверху вниз: верхний уровень оптимизирует «прогресс к цели»
и выдаёт подцель, средний превращает подцель в подцель пониже, нижний —
в конкретные действия.

Ключевой результат статьи: на Visual AntMaze трёхуровневая иерархия поднимает
успех с 18% до 73% при МЕНЬШЕМ бюджете планирования, чем плоский JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .jepa import JEPA, JEPAConfig
from .predictor import context_windows
from ..losses.auxiliary import metric_loss


@dataclass
class HierarchyConfig:
    levels: int = 3
    strides: list[int] = field(default_factory=lambda: [1, 4, 16])
    base: JEPAConfig = field(default_factory=JEPAConfig)


class HJEPA(nn.Module):
    """Один общий энкодер + отдельный предиктор на каждый временной масштаб.

    Общий энкодер — не случайность: по теореме о платоновских представлениях
    (arXiv:2610.39302) разные модальности и масштабы сходятся к одному
    представлению с точностью до поворота. Держать одно пространство дешевле
    и позволяет сравнивать подцели между уровнями напрямую.
    """

    def __init__(self, cfg: HierarchyConfig) -> None:
        super().__init__()
        if cfg.levels != len(cfg.strides):
            raise ValueError("levels должен совпадать с длиной strides")
        self.cfg = cfg
        base = cfg.base
        self.encoder = JEPA(base)
        # Уровни используют веса энкодера, но свои предикторы.
        self.predictors = nn.ModuleList(
            [
                type(self.encoder.predictor)(
                    dim=base.proj_dim,
                    action_dim=base.action_dim,
                    hidden=base.pred_hidden,
                    depth=base.pred_depth,
                    ctx=base.ctx,
                    horizon=1,
                )
                for _ in range(cfg.levels)
            ]
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(obs)

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_sequence(obs)

    def predict(self, level: int, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictors[level](z, action)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Суммарная потеря по всем уровням.

        obs: (B,T,C,H,W). Для уровня со stride s нужен горизонт s, поэтому
        используется только часть последовательности (T >= max(strides)+1).
        """
        z = self.encode_sequence(obs)
        t = z.shape[1]
        ctx = self.cfg.base.ctx
        windows = context_windows(z, ctx)  # (B,T,ctx,D)
        losses = []
        out: dict[str, torch.Tensor] = {}
        for i, s in enumerate(self.cfg.strides):
            if t < s + 1:
                continue
            # Цель уровня i: эмбеддинг через s шагов.
            target = z[:, s:]
            # Вход: окно, заканчивающееся в начале окна длиной s.
            src = windows[:, : t - s]  # (B, t-s, ctx, D)
            # Действие-«макро»: среднее действие за окно длиной s.
            a = actions.unfold(1, s, 1).mean(dim=-1)  # (B, t-s, A)
            nb = src.shape[0] * src.shape[1]
            pred = self.predictors[i](
                src.reshape(nb, ctx, src.shape[-1]), a.reshape(nb, -1)
            )
            loss = nn.functional.smooth_l1_loss(pred, target.reshape(nb, -1))
            losses.append(loss)
            out[f"pred_loss_l{i}"] = loss.detach()
        if not losses:
            raise ValueError("последовательность слишком короткая для иерархии")
        pred_loss = torch.stack(losses).mean()
        reg_loss = self.encoder.sigreg(z.float())

        # Та же метрическая регуляризация, что и в JEPA: без неё латент
        # остаётся «высокочастотным кодом» и планировщик оптимизирует шум.
        met_loss = (
            metric_loss(z, positions)
            if positions is not None
            else z.new_zeros(())
        )

        total = (
            (1.0 - self.cfg.base.lamb) * pred_loss
            + self.cfg.base.lamb * reg_loss
            + self.cfg.base.metric_weight * met_loss
        )
        out.update(
            loss=total,
            pred_loss=pred_loss.detach(),
            metric_loss=met_loss.detach(),
            sigreg_loss=reg_loss.detach(),
        )
        return out
