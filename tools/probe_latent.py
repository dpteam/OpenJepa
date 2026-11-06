"""Диагностика латента: годится ли он для планирования.

    python tools/probe_latent.py --ckpt runs/toy/best.pt

Три проверки, каждая ловит свой класс поломки:

1. **Асимметрия осей.** Кодирует ли латент все управляемые факторы? Если
   сдвиг по X меняет латент в 10 раз слабее, чем по Y, — энкодер выбросил
   горизонталь, и планировщик не сможет двигать объект вбок.

2. **Гладкость.** Меняется ли латент плавно при малом изменении входа?
   Если сдвиг на 0.5 px меняет латент на 2.5, а на 40 px — на 18, то
   «сигнал/шум» латента ужасен: предиктор не может предсказать следующий
   латент, а планировщик оптимизирует шум. Это самая частая и самая
   незаметная поломка: `pred_loss` при этом выглядит приемлемо.

3. **Отношение сигнал/шум.** Отношение среднего изменения латента на 1 px
   к его изменению на 0.5 px. У гладкого кода они отличаются в ~2 раза,
   у шумного — почти не отличаются.

    python tools/probe_latent.py --ckpt runs/toy/best.pt --quick
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


def check_axes(eng, env) -> tuple[float, bool]:
    lo = env.cfg.radius * 2.5
    hi = env.cfg.size - env.cfg.radius * 2.5
    mid = env.cfg.size / 2
    pts = [(lo, lo), (lo, hi), (hi, lo), (hi, hi), (mid, mid),
           (mid, lo), (mid, hi), (lo, mid), (hi, mid)]
    zs = [encode_at(eng, env, p) for p in pts]

    print(f"{'пара':<22} {'латент':>8} {'пиксели':>8} {'ratio':>7}")
    rx, ry = [], []
    for i, j in itertools.combinations(range(len(pts)), 2):
        dx = abs(pts[i][0] - pts[j][0])
        dy = abs(pts[i][1] - pts[j][1])
        if dx and dy:
            continue
        ld = float((zs[i] - zs[j]).norm())
        pd = float(np.hypot(dx, dy))
        r = ld / pd
        (rx if dx else ry).append(r)
        print(f"{str(pts[i]) + '-' + str(pts[j]):<22} {ld:8.3f} {pd:8.1f} {r:7.3f}")

    mx, my = float(np.mean(rx)), float(np.mean(ry))
    print(f"\nсредний ratio по X: {mx:.3f}")
    print(f"средний ratio по Y: {my:.3f}")
    if min(mx, my) < 1e-6:
        print("FAIL: одна из осей не кодируется вообще")
        return float("inf"), False
    asym = max(mx, my) / min(mx, my)
    print(f"асимметрия: {asym:.2f}x")
    ok = asym <= 3.0
    print("OK: латент отражает обе оси" if ok else
          "FAIL: латент игнорирует одну из осей — нужен inverse-dynamics")
    return asym, ok


def check_smoothness(eng, env, y: float = 32.0) -> tuple[float, bool]:
    """Сравнивает изменение латента на 0.5 px и на 2 px."""
    xs = np.arange(24.0, 40.0, 0.5)
    zs = [encode_at(eng, env, (x, y)) for x in xs]
    steps = [float((zs[i] - zs[i - 1]).norm()) for i in range(1, len(zs))]
    # соседние сэмплы: 0.5 px; через 4 сэмпла: 2.0 px
    small = float(np.mean(steps))
    big = float(np.mean([float((zs[i] - zs[i - 4]).norm()) for i in range(4, len(zs))]))

    z_norm = float(zs[0].norm())
    print(f"изменение на 0.5 px: {small:.3f}")
    print(f"изменение на 2.0 px: {big:.3f}")
    print(f"отношение (2px / 0.5px): {big / max(small, 1e-9):.2f}  (у гладкого ~4)")
    print(f"норма латента: {z_norm:.2f}")
    print(f"шум/норма: {small / max(z_norm, 1e-9):.3f}  (должно быть < 0.05)")

    ok = (big / max(small, 1e-9)) > 2.5 and (small / max(z_norm, 1e-9)) < 0.05
    print("OK: латент гладкий" if ok else
          "FAIL: латент высокочастотный — предиктор не сможет его предсказать")
    return small / max(z_norm, 1e-9), ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="только гладкость")
    args = ap.parse_args()

    eng = OpenJEPAEngine.from_checkpoint(args.ckpt)
    env = BallEnv(BallEnvConfig(size=args.size), seed=0)

    ok = True
    if not args.quick:
        print("=== 1. асимметрия осей ===")
        _, ok_axes = check_axes(eng, env)
        ok = ok and ok_axes

    print("\n=== 2. гладкость латента ===")
    _, ok_smooth = check_smoothness(eng, env)
    ok = ok and ok_smooth

    print()
    print("ИТОГ:", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()


