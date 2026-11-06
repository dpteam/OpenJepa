"""Голова-проба: латент -> физические величины.

Зачем нужна. Планировщик может считать стоимость двумя способами:

1. **в латенте**: `‖z_t − z_goal‖²`. Требует, чтобы латент был метрикой
   (см. `research/notes/2026-11-06-sigreg-vs-metric.md`), и чтобы цель
   задавалась состоянием, а не кадром.
2. **в физическом пространстве**: `‖p̂(z_t) − p_goal‖²`, где `p̂` — эта
   проба.

Второй способ честнее для игрушки: цель — это «мяч в точке X», а не «мяч в
точке X с нулевой скоростью». Проба — это **readout** (линейное или почти
линейное отображение), а не декодер картинки: мы по-прежнему не
восстанавливаем пиксели.

Это тот же приём, что probing в LeJEPA: если информация в латенте есть, она
линейно извлекается. Если проба не может предсказать позицию — латент её не
содержит, и это диагноз, а не проблема пробы.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionProbe(nn.Module):
    """z -> p̂. Маленький MLP: сложнее линейного, но не декодер."""

    def __init__(self, dim: int, out_dim: int = 2, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU(approximate="tanh")]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        # Нулевая инициализация выхода: на старте проба предсказывает 0,
        # и её потеря ~O(1) (позиции нормированы в [-1,1]). Иначе случайный
        # линейный слой на латенте с нормой ~5 даёт потерю ~20 и доминирует
        # над остальными членами.
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape[:-1]
        out = self.net(z.reshape(-1, z.shape[-1]))
        return out.reshape(*shape, out.shape[-1])
