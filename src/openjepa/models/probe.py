"""Пространственная проба: сетка токенов -> координаты объекта.

Зачем не обычный MLP на пуле. `ViTEncoder` усредняет токены патчей
(`mean(dim=1)`), чтобы получить один вектор на кадр. Усреднение удобно для
динамики, но **стирает информацию о том, где находится объект**: мяч слева
и мяч справа дают похожие средние активации.

Замер на нашем чекпоинте: MLP-проба на пуле сходилась к ошибке **11 px** на
сцене 64 px — это 17% и бесполезно для планирования.

Решение — **spatial soft-argmax**, стандартный приём для чтения координат
(используется в позе человека, трекинге точек, ключевых точках):

1. каждый токен патча проецируется в один скаляр (логит «здесь объект»);
2. softmax по всем патчам даёт распределение внимания;
3. математическое ожидание координат сетки под этим распределением —
   искомая точка.

Преимущества:

* координата получается **непрерывной** (взвешенное среднее), а не
  квантованной по патчам;
* суб-пиксельная точность достигается даже при `patch_size=8`;
* нет декодера пикселей — это readout признаков, а не реконструкция;
* градиент течёт и в пробу, и в энкодер, поэтому энкодер вынужден сохранять
  пространственную информацию.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialProbe(nn.Module):
    """Сетка токенов (..., N, D) -> координаты (..., out_dim) в [-1, 1].

    Args:
        dim: размерность токена.
        grid: размер сетки патчей (`img_size // patch_size`).
        out_dim: 2 для (x, y); можно 3 для (x, y, z).
        hidden: ширина проекции токена в логит.
        temperature: «резкость» softmax. Больше — острее.
    """

    def __init__(
        self,
        dim: int,
        grid: int = 8,
        out_dim: int = 2,
        hidden: int = 64,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.out_dim = out_dim
        self.temperature = temperature

        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, 1),
        )

        # Фиксированные координаты центров патчей в [-1, 1].
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid),
            torch.linspace(-1.0, 1.0, grid),
            indexing="ij",
        )
        coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (N, 2)
        if out_dim > 2:
            coords = torch.cat([coords, torch.zeros(coords.shape[0], out_dim - 2)], dim=-1)
        self.register_buffer("coords", coords, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (..., N, D) -> (..., out_dim).

        Мягкое внимание: если объект один, распределение почти одноточечное,
        и ответ — его центр. Если объектов несколько, ответ — «центр масс»,
        что для планирования часто и нужно.
        """
        if tokens.ndim < 3:
            raise ValueError(f"ожидался (...,N,D), получено {tuple(tokens.shape)}")
        n = tokens.shape[-2]
        if n != self.coords.shape[0]:
            raise ValueError(
                f"число токенов {n} не совпадает с сеткой {self.coords.shape[0]}"
            )
        logits = self.score(tokens).squeeze(-1)  # (..., N)
        attn = torch.softmax(logits / self.temperature, dim=-1)
        return attn @ self.coords.to(tokens.dtype)  # (..., out_dim)

    def attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """Распределение внимания по патчам — для визуализации и отладки."""
        logits = self.score(tokens).squeeze(-1)
        return torch.softmax(logits / self.temperature, dim=-1)


class PositionSlice(nn.Module):
    """Берёт последние `out_dim` компонент латента как координаты.

    Используется, когда позиция **встроена в латент** (`position_in_latent`):
    предиктор учится предсказывать её вместе с остальным, а планировщику
    остаётся просто взять эти компоненты. Никакого обучения не нужно.

    Почему так лучше MLP-пробы: mean-pooling в энкодере стирает позицию
    (ошибка 11–14 px), и MLP не может вытащить то, чего нет. Если позиция
    лежит в латенте явно, планировщик получает её с точностью пробега
    предиктора, а не с точностью угадывания.
    """

    def __init__(self, out_dim: int = 2) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., -self.out_dim :]


class PositionProbe(nn.Module):
    """Старая проба на пуле: z -> p. Оставлена для сравнения.

    На практике упирается в ~11 px, потому что mean-pooling стирает
    пространственную информацию. Используйте `SpatialProbe`.
    """

    def __init__(self, dim: int, out_dim: int = 2, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU(approximate="tanh")]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(approximate="tanh")]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        shape = z.shape[:-1]
        out = self.net(z.reshape(-1, z.shape[-1]))
        return out.reshape(*shape, out.shape[-1])
