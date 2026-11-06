"""Демо: прогоняет планирование и сохраняет полосу кадров + GIF.

    python tools/demo.py --ckpt runs/toy/best.pt --out runs/demo

Пишет:
  runs/demo/strip.png   — 8 кадров из эпизода
  runs/demo/traj.json   — позиции, действия, surprise
  runs/demo/README.md   — краткий отчёт по эпизоду

Не требует matplotlib. GIF собирается через PIL, если он установлен.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.data.synthetic import BallEnv, BallEnvConfig  # noqa: E402
from openjepa.runtime.engine import OpenJEPAEngine, RuntimeConfig, make_goal_frame  # noqa: E402


def save_strip(frames: list[np.ndarray], path: Path, n: int = 8) -> None:
    from torchvision.utils import save_image

    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    imgs = [torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0 for i in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(imgs), path, nrow=n)


def save_gif(frames: list[np.ndarray], path: Path, scale: int = 3) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    imgs = [Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale),
                                     Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    eng = OpenJEPAEngine.from_checkpoint(
        args.ckpt,
        RuntimeConfig(plan_horizon=args.horizon, plan_samples=args.samples),
    )
    env = BallEnv(BallEnvConfig(), seed=args.seed)
    frame = env.reset()
    goal_pos = np.array([env.cfg.size * 0.78, env.cfg.size * 0.25])
    goal_frame = make_goal_frame(env, goal_pos)

    eng.reset()
    eng.set_goal(goal_frame)

    frames = [frame.copy()]
    traj = {"pos": [env.pos.tolist()], "action": [], "surprise": [], "goal": goal_pos.tolist()}
    for _ in range(args.steps):
        action, surprise = eng.step(frame)
        frame = env.step(action)
        frames.append(frame.copy())
        traj["pos"].append(env.pos.tolist())
        traj["action"].append(action.tolist())
        traj["surprise"].append(surprise)

    dist = float(np.linalg.norm(env.pos - goal_pos))
    traj["final_dist"] = dist

    save_strip(frames, out / "strip.png")
    (out / "traj.json").write_text(json.dumps(traj, indent=2), encoding="utf-8")
    gif_ok = save_gif(frames, out / "rollout.gif")

    report = [
        f"# Демо: {args.ckpt}",
        "",
        f"* цель: `{goal_pos.round(1).tolist()}`",
        f"* финальная позиция: `{env.pos.round(1).tolist()}`",
        f"* дистанция: **{dist:.2f} px** (порог успеха {env.cfg.radius * 3:.0f} px)",
        f"* шагов: {args.steps}",
        f"* средний surprise: {np.mean(traj['surprise']):.4f}",
        "",
        "Файлы: `strip.png` (8 кадров), `rollout.gif`"
        + ("" if gif_ok else " (нет PIL)") + ", `traj.json`.",
    ]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")

    print(f"дистанция {dist:.2f} px | средний surprise {np.mean(traj['surprise']):.4f}")
    print(f"сохранено в {out}")


if __name__ == "__main__":
    main()
