"""Матрица экспериментов: что делает латент пригодным для планирования.

    python tools/sweep.py --out runs/sweep

Главная метрика — не `pred_loss`, а **open-loop ошибка на 8 шагов**.
Планировщик «воображает» 6–8 шагов; если ошибка там больше нормы латента,
он оптимизирует шум, и успех падает независимо от того, как красиво
выглядит кривая обучения.

Для каждого варианта: короткое обучение, затем замеры:

    pred1   — ошибка предсказания на 1 шаг (eval)
    open8   — средняя ошибка open-loop rollout на 8 шагов (главная)
    ratio   — open8 / ‖z‖ (должно быть < 0.5)
    asym    — асимметрия латента по X/Y (должна быть < 3)
    noise   — изменение латента на 0.5 px / ‖z‖ (должно быть < 0.05)
    succ    — успех CEM-планирования
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openjepa.data.synthetic import (  # noqa: E402
    BallEnv,
    BallEnvConfig,
    smooth_random_actions,
)
from openjepa.planning.cem import CEMConfig, CEMPlanner  # noqa: E402
from openjepa.train.pretrain import train  # noqa: E402


VARIANTS: dict[str, dict] = {
    # базовая линия: без размытия, D=64
    "base": {},
    # меньше размерность латента: меньше «лишних» измерений под шум
    "proj8": dict(proj_dim=8),
    # размытие входа: делает сигнал плавным по построению
    "blur2": dict(env_blur=2.0),
    "blur2_proj8": dict(env_blur=2.0, proj_dim=8),
    "blur2_proj16": dict(env_blur=2.0, proj_dim=16),
    "blur2_proj8_nometric": dict(env_blur=2.0, proj_dim=8, metric_weight=0.0),
}


def make_cfg(over: dict, epochs: int) -> dict:
    over = dict(over)
    env_blur = over.pop("env_blur", 0.0)
    base = {
        "seed": 0,
        "device": "auto",
        "out_dir": "runs/sweep",
        "data": {
            "n_traj": 1200,
            "n_val": 64,
            "length": 12,
            "env": {"size": 64, "radius": 5.0, "blur": env_blur},
        },
        "model": {
            "img_size": 64,
            "patch_size": 8,
            "in_chans": 3,
            "dim": 192,
            "depth": 6,
            "heads": 3,
            "proj_dim": 64,
            "proj_hidden": 512,
            "action_dim": 2,
            "ctx": 2,
            "horizon": 1,
            "pred_depth": 3,
            "pred_hidden": 384,
            "lamb": 0.05,
            "inv_weight": 0.5,
            "metric_weight": 1.0,
            "sigreg_slices": 256,
            "hierarchical": False,
        },
        "train": {
            "epochs": epochs,
            "batch_size": 32,
            "eval_batch_size": 32,
            "lr": 3e-4,
            "weight_decay": 0.05,
            "warmup_ratio": 0.05,
            "amp": True,
            "workers": 0,
            "save_every": 0,
            "grad_clip": 0,
        },
    }
    base["model"].update(over)
    return base


# --------------------------------------------------------------------------- #
def measure(ckpt: Path) -> dict:
    from openjepa.runtime.engine import OpenJEPAEngine

    eng = OpenJEPAEngine.from_checkpoint(ckpt)
    m, dev = eng.model, eng.device
    ctx = m.cfg.ctx
    # среду берём с теми же параметрами, с какими обучали (в частности blur)
    env_cfg = BallEnvConfig(**eng.cfg.get("data", {}).get("env", {}))

    env = BallEnv(env_cfg, seed=77)

    # --- open-loop rollout ------------------------------------------------- #
    rng = np.random.default_rng(77)
    f0 = env.reset()
    acts = smooth_random_actions(rng, 10, dim=2)
    frames = [f0]
    for a in acts:
        frames.append(env.step(a))
    arr = np.stack(frames)
    x = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    x = ((x - 0.5) / 0.5).unsqueeze(0).to(dev)

    with torch.no_grad():
        Z = m.encode_sequence(x)[0]
        A = torch.from_numpy(acts).float().to(dev).unsqueeze(0)
        hist = Z[:ctx].unsqueeze(0)
        errs = []
        for t in range(8):
            p = m.predict_next(hist, A[:, t])
            errs.append(float((p - Z[t + ctx]).norm()))
            hist = torch.cat([hist[:, 1:], p.unsqueeze(1)], dim=1)
        pred1, open8 = errs[0], float(np.mean(errs))
        znorm = float(Z.norm(dim=-1).mean())

    # --- асимметрия и шумность --------------------------------------------- #
    def z_at(pos):
        env.pos = np.asarray(pos, dtype=np.float64).copy()
        env.vel = np.zeros(2)
        return eng.encode(env.render()).squeeze(0)

    lo, hi, mid = 12.5, 51.5, 32.0
    zx = float((z_at((lo, mid)) - z_at((hi, mid))).norm())
    zy = float((z_at((mid, lo)) - z_at((mid, hi))).norm())
    asym = max(zx, zy) / max(min(zx, zy), 1e-6)

    xs = np.arange(26.0, 38.0, 0.5)
    zs = [z_at((x, mid)) for x in xs]
    steps = [float((zs[i] - zs[i - 1]).norm()) for i in range(1, len(zs))]
    noise = float(np.mean(steps)) / max(znorm, 1e-9)

    # --- планирование ------------------------------------------------------ #
    # Используем планировщик движка: он сам выбирает режим стоимости
    # (probe, если у модели есть проба, иначе latent).
    eng.rt.plan_horizon = 6
    eng.rt.plan_samples = 128
    eng.rt.plan_iters = 2
    eng.planner.cfg.horizon = 6
    eng.planner.cfg.n_samples = 128
    eng.planner.cfg.iters = 2

    ok, dists = 0, []
    rng2 = np.random.default_rng(11)
    for _ in range(6):
        gp = rng2.uniform(lo, hi, size=2)
        frame = env.reset()
        eng.reset()
        if eng.cost_mode.value == "probe":
            eng.set_goal(gp)          # позиция в пикселях, движок нормирует
        else:
            eng.set_goal(z_at(gp))
        for _ in range(25):
            act, _ = eng.step(frame)
            frame = env.step(act)
        d = float(np.linalg.norm(env.pos - gp))
        dists.append(d)
        ok += int(d < env.cfg.radius * 3)

    return {
        "pred1": pred1,
        "open8": open8,
        "ratio": open8 / max(znorm, 1e-9),
        "asym": asym,
        "noise": noise,
        "succ": ok / 6,
        "dist": float(np.mean(dists)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--only", default=None, help="через запятую")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(VARIANTS)

    rows = []
    for name in names:
        if name not in VARIANTS:
            print(f"нет варианта {name}, пропускаю")
            continue
        print(f"\n=== {name} ===", flush=True)
        cfg = make_cfg(VARIANTS[name], args.epochs)
        run_dir = out / name
        try:
            train(cfg, run_dir)
            res = measure(run_dir / "best.pt")
            res["name"] = name
            rows.append(res)
            print(
                f"  pred1={res['pred1']:.2f} open8={res['open8']:.2f} "
                f"ratio={res['ratio']:.2f} asym={res['asym']:.2f} "
                f"noise={res['noise']:.3f} succ={res['succ']:.0%}",
                flush=True,
            )
        except Exception as exc:
            print(f"  ОШИБКА: {exc}", flush=True)
            rows.append({"name": name, "error": str(exc)})

    (out / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"{'вариант':<14}{'pred1':>7}{'open8':>7}{'ratio':>7}{'asym':>7}{'noise':>8}{'succ':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<14}  ошибка: {r['error'][:40]}")
            continue
        print(
            f"{r['name']:<14}{r['pred1']:>7.2f}{r['open8']:>7.2f}{r['ratio']:>7.2f}"
            f"{r['asym']:>7.2f}{r['noise']:>8.3f}{r['succ']:>6.0%}"
        )


if __name__ == "__main__":
    main()
