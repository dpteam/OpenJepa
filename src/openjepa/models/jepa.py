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
from .probe import PositionProbe
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
            dim=cfg.proj_dim,
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
        self.inverse = InverseDynamics(cfg.proj_dim, cfg.action_dim, cfg.pred_hidden)
        # Проба латент -> физические координаты. Нужна, чтобы планировщик мог
        # считать стоимость в пространстве задачи, а не в латенте
        # (см. research/notes/2026-11-06-goal-definition.md).
        self.probe = PositionProbe(cfg.proj_dim, out_dim=2, hidden=128)

    # ------------------------------------------------------------------ #
    # inference helpers
    # ------------------------------------------------------------------ #
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,Dp) в пространстве предсказания."""
        return self.projector(self.encoder(obs))

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """(B,T,C,H,W) -> (B,T,Dp)."""
        return self.projector(self.encoder(obs))

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

        z = self.encode_sequence(obs)  # (B,T,Dp)
        ctx = self.cfg.ctx
        if t < ctx + 1:
            raise ValueError(f"нужно >= {ctx + 1} кадров при ctx={ctx}")

        # Окна: (B,T,ctx,Dp). Окно t заканчивается на z_t.
        windows = context_windows(z, ctx)

        if self.cfg.horizon > 1:
            k = self.cfg.horizon
            if actions.shape[1] < k:
                raise ValueError(f"нужно >= {k} действий при horizon={k}")
            # Стартовое окно одно, действий — чанк из k, целей — k.
            pred = self.predictor(windows[:, 0], actions[:, :k])
            target = z[:, 1 : k + 1]
        else:
            src = windows[:, :-1]  # (B,T-1,ctx,Dp)
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

        # SIGReg считаем в fp32: статистика — разность близких величин
        # (cos_mean - phi), в bf16 она теряет точность и «шумит».
        reg_loss = self.sigreg(z.float())

        # Метрическая регуляризация: латентное расстояние должно быть
        # пропорционально реальному, иначе планировщик оптимизирует шум.
        # Форма безразмерная (коэффициент вариации), см. losses/auxiliary.py.
        met_loss = metric_loss(z, positions) if positions is not None else z.new_zeros(())

        # Проба: латент должен линейно (почти) содержать физические координаты.
        # Это одновременно (а) даёт планировщику стоимость в пространстве
        # задачи и (б) служит диагностикой: если проба не учится, латент
        # не содержит нужной информации.
        probe_loss = z.new_zeros(())
        if positions is not None and self.cfg.probe_weight > 0:
            p_hat = self.probe(z.float())
            probe_loss = F.smooth_l1_loss(p_hat, positions.float())

        # Защита от седловой точки SIGReg в нуле. Нужна, если проектор не
        # использует BatchNorm (см. losses/auxiliary.py::variance_floor).
        var_loss = variance_floor(z, self.cfg.var_target)

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
