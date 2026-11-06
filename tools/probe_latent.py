"""Диагностика латента: кодирует ли он то, чем можно управлять.

    python tools/probe_latent.py --ckpt runs/toy/best.pt

Зачем: `pred_loss` может быть низким, а латент — бесполезным для планирования.
Классический случай — энкодер выучил только те факторы, которые легко
предсказать (например, вертикаль под гравитацией), и выбросил те, на которые
влияет действие (горизонталь). Планировщик тогда физически не может двигать
объект в нужную сторону.

Скрипт измеряет, насколько латентное расстояние отражает изменение каждого
управляемого фактора, и сравнивает с «идеальным» (пропорциональным).

    ratio_x ≈ ratio_y  →  латент использует оба фактора
    ratio_x ≪ ratio_y  →  горизонталь потеряна, нужен inverse-dynamics
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.data.synthetic import BallEnv, BallEnvConfig  # noqa: E402
from openjepa.runtime.engine import OpenJEPAEngine  # noqa: E402


def encode_at(eng: OpenJEPAEngine, env: BallEnv, pos) -> torch.Tensor:
    env.pos = np.asarray(pos, dtype=np.float64).copy()
    env.vel = np.zeros(2)
    return eng.encode(env.render()).squeeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()

    eng = OpenJEPAEngine.from_checkpoint(args.ckpt)
    env = BallEnv(BallEnvConfig(size=args.size), seed=0)

    lo = env.cfg.radius * 2.5
    hi = env.cfg.size - env.cfg.radius * 2.5
    mid = env.cfg.size / 2
    pts = [
        (lo, lo), (lo, hi), (hi, lo), (hi, hi), (mid, mid),
        (mid, lo), (mid, hi), (lo, mid), (hi, mid),
    ]
    zs = [encode_at(eng, env, p) for p in pts]

    print(f"{'пара':<22} {'латент':>8} {'пиксели':>8} {'ratio':>7}")
    ratios_x, ratios_y = [], []
    for i, j in itertools.combinations(range(len(pts)), 2):
        dx = abs(pts[i][0] - pts[j][0])
        dy = abs(pts[i][1] - pts[j][1])
        if dx and dy:
            continue  # диагонали пропускаем, они смешивают оси
        ld = float((zs[i] - zs[j]).norm())
        pd = float(np.hypot(dx, dy))
        r = ld / pd
        axis = "x" if dx else "y"
        (ratios_x if dx else ratios_y).append(r)
        print(f"{str(pts[i])+'-'+str(pts[j]):<22} {ld:8.3f} {pd:8.1f} {r:7.3f}  ({axis})")

    mx = float(np.mean(ratios_x))
    my = float(np.mean(ratios_y))
    print(f"\nсредний ratio по X: {mx:.3f}")
    print(f"средний ratio по Y: {my:.3f}")
    if mx < 1e-6 or my < 1e-6:
        print("FAIL: одна из осей не кодируется вообще")
        raise SystemExit(1)
    asym = max(mx, my) / min(mx, my)
    print(f"асимметрия: {asym:.2f}x")
    if asym > 3.0:
        print("FAIL: латент игнорирует одну из осей — нужен inverse-dynamics")
        raise SystemExit(1)
    print("OK: латент отражает обе оси")


if __name__ == "__main__":
    main()
