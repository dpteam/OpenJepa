"""Минимальный ViT-энкодер: картинки и видео -> один вектор на кадр.

Почему ViT, а не ResNet: JEPA-литература (I-JEPA, V-JEPA, LeJEPA) показывает,
что ViT лучше держит абстракцию при маскировании и предсказании в латентном
пространстве, и на нём же проще делать иерархию по времени.

Энкодер намеренно без CLS-токена: берём среднее по патчам. Для world model
это устойчивее (нет «мусорного» токена, который легко схлопывается).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Conv-патчи + обучаемый позиционный эмбеддинг с интерполяцией по размеру."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} должно делиться на patch_size={patch_size}"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, g, g)
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, gh*gw, D)
        if (gh, gw) != (self.grid, self.grid):
            pos = self.pos.transpose(1, 2).reshape(1, d, self.grid, self.grid)
            pos = F.interpolate(pos, size=(gh, gw), mode="bicubic", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos
        return x + pos


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim должно делиться на heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(
            2, 0, 3, 1, 4
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Вход (B,C,H,W) или (B,T,C,H,W). Выход (B,D) или (B,T,D)."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 4
        if squeeze:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"ожидался (B,C,H,W) или (B,T,C,H,W), получено {x.shape}")
        b, t, c, h, w = x.shape
        z = self.patch_embed(x.reshape(b * t, c, h, w))
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).mean(dim=1).reshape(b, t, -1)
        return z.squeeze(1) if squeeze else z


class Projector(nn.Module):
    """MLP-голова: приводит эмбеддинг к размерности, на которой живёт SIGReg.

    Отдельный projector нужен, потому что «широкий» признак энкодера и
    «узкое» пространство предсказания — разные вещи. В I-JEPA/V-JEPA это
    стандартная практика.

    Нормировка — **BatchNorm по умолчанию**, и это не «эвристика ради
    эвристики». Причина: у SIGReg есть седловая точка в `z = 0`. Если сеть
    может бесплатно уменьшить ошибку предсказания, сжимая латент к нулю,
    она это сделает — и статистика Эппса–Пули в этой точке имеет нулевой
    градиент, то есть вытащить модель оттуда уже нельзя. BN нормирует
    признаки **по батчу**, поэтому «все сэмплы одинаковые» становится
    недостижимым состоянием. Ровно так же устроена референсная реализация
    LeJEPA (torchvision `MLP(..., norm_layer=BatchNorm1d)`).

    Альтернатива `norm="layer"` есть для случаев, когда батч равен 1 при
    обучении, но она не защищает от коллапса — проверено.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, norm: str = "batch"):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
        if norm == "batch":
            layers.append(nn.BatchNorm1d(hidden))
        elif norm == "layer":
            layers.append(nn.LayerNorm(hidden))
        elif norm != "none":
            raise ValueError(f"неизвестная нормировка: {norm}")
        layers += [nn.GELU(), nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        self.norm_kind = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = self.net(x.reshape(-1, shape[-1]))
        return y.reshape(*shape[:-1], y.shape[-1])
