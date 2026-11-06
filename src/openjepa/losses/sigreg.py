"""SIGReg — Sketched Isotropic Gaussian Regularization.

Ключевая идея (Balestriero & LeCun, arXiv:2511.08544):

    Эмбеддинги энкодера должны быть распределены как изотропный гауссиан.
    Тогда (а) представление не коллапсирует и (б) линейная/нелинейная
    проба на любой downstream-задаче имеет минимальный worst-case риск.

Как это проверяется: распределение в R^D сравнивается с N(0, I) не напрямую
(проклятие размерности), а через M случайных одномерных проекций. Для каждой
проекции берётся характеристическая функция (CF) эмпирического распределения
и сравнивается с CF стандартного гауссиана в квадратичной норме с гауссовым
ядром — это статистика Эппса–Пули.

    EP(a) = N * ∫ |φ̂_a(t) - e^{-t²/2}|² e^{-t²/2} dt

Интеграл считается квадратурой по сетке t. Для стандартного гауссиана
φ(t) = e^{-t²/2}, поэтому |φ̂ - φ|² = (E cos(tz) - e^{-t²/2})² + (E sin(tz))².

Свойства, ради которых это используется вместо VICReg/BN/stop-grad:
  * один гиперпараметр (вес λ в общей потере);
  * O(N) по времени и памяти;
  * ограниченные градиенты (теорема 4 в статье), поэтому обучение не взрывается;
  * не нужен teacher-student, EMA, whitening, stop-gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch-based goodness-of-fit test against an isotropic Gaussian.

    Args:
        knots: число узлов квадратуры по t (17 — значение из статьи).
        t_max: верхняя граница интегрирования. 3.0 покрывает основную массу
            гауссова ядра e^{-t²/2}.
        num_slices: число случайных направлений M (256 в референсе).
        generator: опциональный torch.Generator для воспроизводимости.

    Shape:
        вход: (..., D) — любые ведущие размерности, D = размер эмбеддинга.
        выход: скаляр (усреднение по всем ведущим размерностям).
    """

    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        num_slices: int = 256,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.generator = generator

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        # Трапеции: края с половинным весом.
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)

        # Интегрируем по [0, t_max] и удваиваем: подынтегральное выражение чётное.
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"SIGReg ожидает (..., D), получено {tuple(x.shape)}")

        dim = x.shape[-1]
        flat = x.reshape(-1, dim)
        n = flat.shape[0]
        if n < 2:
            return flat.new_zeros(())

        # Случайные единичные направления. Регенерируются каждый шаг — это
        # часть «скетчинга»: разные направления на разных итерациях дают
        # несмещённую оценку статистики.
        if self.generator is not None:
            a = torch.randn(
                dim, self.num_slices, generator=self.generator, device="cpu"
            ).to(flat.device)
        else:
            a = torch.randn(dim, self.num_slices, device=flat.device, dtype=flat.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        # (N, M, K): проекции * узлы квадратуры
        proj = flat @ a
        x_t = proj.unsqueeze(-1) * self.t.to(flat.dtype)

        # Эмпирическая CF по N сэмплам: (M, K)
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)

        err = (cos_mean - self.phi.to(flat.dtype)).square() + sin_mean.square()
        statistic = (err @ self.weights.to(flat.dtype)) * n
        return statistic.mean()


def sigreg(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Функциональная обёртка (создаёт модуль на каждый вызов — для тестов)."""
    return SIGReg(**kwargs)(x)
