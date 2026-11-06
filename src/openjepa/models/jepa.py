"""Сборка JEPA-модели: энкодер + проектор + предиктор + функция потерь.

Формула обучения (LeJEPA, arXiv:2511.08544):

    L = (1 - λ) * L_pred + λ * SIGReg(Z)

где L_pred — ошибка предсказания эмбеддинга следующего наблюдения, а SIGReg
держит распределение эмбеддингов изотропно-гауссовым. Никаких stop-gradient,
EMA-учителей и расписаний: λ — единственный гиперпараметр баланса.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.sigreg import SIGReg
from ..losses.auxiliary import inverse_dynamics_loss, metric_loss, variance_floor
from .predictor import InverseDynamics, LatentPredictor, context_windows
from .probe import PositionProbe, PositionSlice, SpatialProbe
from .vit import Projector, ViTEncoder


@dataclass
class JEPAConfig:
    # восприятие
    img_size: int = 64
    patch_size: int = 8
    in_chans: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 3
    # латентное пространство предсказания
    proj_dim: int = 64
    proj_hidden: int = 512
    proj_norm: str = "batch"
    # динамика
    action_dim: int = 2
    ctx: int = 2
    horizon: int = 1
    pred_depth: int = 3
    pred_hidden: int = 384
    # потеря
    lamb: float = 0.05
    inv_weight: float = 0.5
    metric_weight: float = 1.0
    probe_weight: float = 1.0
    var_weight: float = 1.0
    var_target: float = 1.0
    position_in_latent: bool = True
    stop_grad_target: bool = False
    sigreg_slices: int = 256
    # прочее
    extra: dict = field(default_factory=dict)


class JEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
        )
        self.projector = Projector(
            cfg.dim, cfg.proj_hidden, cfg.proj_dim, norm=cfg.proj_norm
        )
        self.predictor = LatentPredictor(
            dim=self.latent_dim,
            action_dim=cfg.action_dim,
            hidden=cfg.pred_hidden,
            depth=cfg.pred_depth,
            ctx=cfg.ctx,
            horizon=cfg.horizon,
        )
        self.sigreg = SIGReg(num_slices=cfg.sigreg_slices)
        # Вспомогательная голова: (z_t, z_{t+1}) -> a_t. Нужна, чтобы латент
        # не выбрасывал факторы, на которые влияет действие. См. docstring
        # InverseDynamics и docs/14-zhurnal-resheniy.md §8.
        self.inverse = InverseDynamics(self.latent_dim, cfg.action_dim, cfg.pred_hidden)
        # Проба читает координаты из СЕТКИ ТОКЕНОВ (spatial soft-argmax):
        # усреднение по патчам стирает позицию. См. models/probe.py.
        self.probe = SpatialProbe(
            cfg.dim, grid=cfg.img_size // cfg.patch_size, out_dim=2
        )
        # Как планировщик читает позицию из латента:
        #  * позиция встроена в латент -> просто срез последних 2 компонент;
        #  * иначе -> обучаемая голова (менее точно: пул теряет позицию).
        self.z_probe = (
            PositionSlice(2) if cfg.position_in_latent
            else PositionProbe(cfg.proj_dim, out_dim=2, hidden=128)
        )

    @property
    def latent_dim(self) -> int:
        """Размерность пространства, в котором работает предиктор."""
        return self.cfg.proj_dim + (2 if self.cfg.position_in_latent else 0)

    # ------------------------------------------------------------------ #
    # inference helpers
    # ------------------------------------------------------------------ #
    def _encode_full(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает (латент предсказания, пул признаков) для входа энкодера."""
        tok, squeeze = self.encoder.forward_tokens(obs)   # (B,T,N,D)
        pooled = self.projector(tok.mean(dim=2))          # (B,T,Dp)
        if self.cfg.position_in_latent:
            p_hat = self.probe(tok.float())               # (B,T,2)
            full = torch.cat([pooled, p_hat], dim=-1)
        else:
            full = pooled
        if squeeze:
            return full.squeeze(1), pooled.squeeze(1)
        return full, pooled

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B, D) в пространстве предсказания."""
        return self._encode_full(obs)[0]

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,T,C,H,W) -> (B,T,D)."""
        return self._encode_full(obs)[0]

    @torch.no_grad()
    def read_position(self, obs: torch.Tensor) -> torch.Tensor:
        """Координаты объекта по наблюдению, в `[-1, 1]`.

        Читаются из сетки токенов через spatial soft-argmax: mean-pooling
        стирает позицию. См. `models/probe.py`.
        """
        tok, _ = self.encoder.forward_tokens(obs)
        return self.probe(tok.float())

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """z: (B,Dp) или (B,ctx,Dp)."""
        return self.predictor(z, action)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Разворачивает предиктор на len(actions) шагов.

        Args:
            z0: (B,Dp) или (B,ctx,Dp) — стартовый латент (или окно).
            actions: (B,H,A).

        Returns:
            (B,H+1,Dp), включая стартовый латент.

        Контекст поддерживается автоматически: после каждого предсказания окно
        сдвигается влево, поэтому «производная» (скорость объекта) сохраняется
        на всём горизонте. Без этого планировщик «воображал» бы мир, в котором
        объекты не имеют инерции.
        """
        ctx = self.cfg.ctx
        if z0.ndim == 2:
            hist = z0.unsqueeze(1).expand(-1, ctx, -1)
        elif z0.ndim == 3:
            hist = z0
        else:
            raise ValueError(f"z0 должен быть (B,D) или (B,ctx,D), получено {z0.shape}")

        zs = [hist[:, -1]]
        for t in range(actions.shape[1]):
            pred = self.predictor(hist, actions[:, t])
            hist = torch.cat([hist[:, 1:], pred.unsqueeze(1)], dim=1)
            zs.append(pred)
        return torch.stack(zs, dim=1)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """obs: (B,T,C,H,W), actions: (B,T-1,A), positions: (B,T,2) опционально.

        `positions` — привилегированная информация из симулятора. Она нужна
        только для метрической регуляризации (см. ниже) и недоступна на
        инференсе.
        """
        if obs.ndim != 5:
            raise ValueError(f"obs должен быть (B,T,C,H,W), получено {tuple(obs.shape)}")
        b, t = obs.shape[:2]
        if t < 2:
            raise ValueError("нужно хотя бы 2 кадра")

        tok, _ = self.encoder.forward_tokens(obs)      # (B,T,N,D)
        pooled = self.projector(tok.mean(dim=2))       # (B,T,Dp)
        if self.cfg.position_in_latent:
            # Позиция — часть латента. Тогда предиктор предсказывает её
            # вместе с остальным, и планировщик читает её напрямую.
            p_hat = self.probe(tok.float())            # (B,T,2)
            z = torch.cat([pooled, p_hat], dim=-1)
        else:
            p_hat = None
            z = pooled
        ctx = self.cfg.ctx
        if t < ctx + 1:
            raise ValueError(f"нужно >= {ctx + 1} кадров при ctx={ctx}")

        # Окна: (B,T,ctx,D). Окно t заканчивается на z_t.
        windows = context_windows(z, ctx)

        if self.cfg.horizon > 1:
            k = self.cfg.horizon
            if actions.shape[1] < k:
                raise ValueError(f"нужно >= {k} действий при horizon={k}")
            # Стартовое окно одно, действий — чанк из k, целей — k.
            pred = self.predictor(windows[:, 0], actions[:, :k])
            target = z[:, 1 : k + 1]
        else:
            src = windows[:, :-1]  # (B,T-1,ctx,D)
            nb = b * (t - 1)
            pred = self.predictor(src.reshape(nb, ctx, -1), actions.reshape(nb, -1))
            target = z[:, 1:].reshape(nb, -1)

        if self.cfg.stop_grad_target:
            target = target.detach()
        pred_loss = F.smooth_l1_loss(pred, target)

        # Inverse dynamics: восстанавливаем действие по паре соседних латентов.
        # Это заставляет энкодер сохранять управляемые факторы (см. класс
        # InverseDynamics). Без этого латент терял горизонтальную позицию.
        inv_loss = inverse_dynamics_loss(self.inverse, z, actions)

        # SIGReg — только на пуле признаков. Позиция в латенте имеет
        # собственную структуру (ограничена квадратом [-1,1]²), и требовать
        # от неё изотропного гауссиана бессмысленно: это конфликтовало бы с
        # самим смыслом этих компонент.
        reg_loss = self.sigreg(pooled.float())

        # Метрическая регуляризация: латентное расстояние должно быть
        # пропорционально реальному, иначе планировщик оптимизирует шум.
        # Считаем на пуле — там геометрия и должна быть осмысленной.
        met_loss = (
            metric_loss(pooled, positions) if positions is not None else z.new_zeros(())
        )

        # Проба: точность чтения координат из токенов и из латента.
        probe_loss = z.new_zeros(())
        if positions is not None and self.cfg.probe_weight > 0:
            pos = positions.float()
            if p_hat is not None:
                probe_loss = F.smooth_l1_loss(p_hat, pos)
            else:
                probe_loss = F.smooth_l1_loss(self.probe(tok.float()), pos) + F.smooth_l1_loss(
                    self.z_probe(pooled.float()), pos
                )

        # Защита от седловой точки SIGReg в нуле. Нужна, если проектор не
        # использует BatchNorm (см. losses/auxiliary.py::variance_floor).
        var_loss = variance_floor(pooled, self.cfg.var_target)

        total = (
            (1.0 - self.cfg.lamb)
            * (pred_loss + self.cfg.inv_weight * inv_loss)
            + self.cfg.lamb * reg_loss
            + self.cfg.metric_weight * met_loss
            + self.cfg.probe_weight * probe_loss
            + self.cfg.var_weight * var_loss
        )
        return {
            "loss": total,
            "pred_loss": pred_loss.detach(),
            "inv_loss": inv_loss.detach(),
            "metric_loss": met_loss.detach(),
            "probe_loss": probe_loss.detach(),
            "var_loss": var_loss.detach(),
            "sigreg_loss": reg_loss.detach(),
        }

    # ------------------------------------------------------------------ #
    def param_groups(self, lr: float, weight_decay: float = 0.05) -> list[dict]:
        """AdamW-группы: без weight decay для норм, сдвигов и позиций."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "pos" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
