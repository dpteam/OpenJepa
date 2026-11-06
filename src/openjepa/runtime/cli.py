"""CLI рантайма: обучение, оценка, демо, HTTP-сервер.

    python -m openjepa.runtime.cli train  --config configs/pretrain_toy.yaml
    python -m openjepa.runtime.cli eval   --ckpt runs/toy/best.pt --episodes 20
    python -m openjepa.runtime.cli demo   --ckpt runs/toy/best.pt --out runs/demo.png
    python -m openjepa.runtime.cli serve  --ckpt runs/toy/best.pt --port 8077
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.synthetic import BallEnv, BallEnvConfig
from .engine import OpenJEPAEngine, RuntimeConfig, make_goal_frame


def _save_strip(frames: list[np.ndarray], path: Path, n: int = 8) -> None:
    from torchvision.utils import save_image

    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    imgs = [torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0 for i in idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.stack(imgs), path, nrow=n)


def cmd_train(args: argparse.Namespace) -> None:
    from ..train.pretrain import load_config, train

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.out)


def cmd_eval(args: argparse.Namespace) -> None:
    eng = OpenJEPAEngine.from_checkpoint(
        args.ckpt,
        RuntimeConfig(
            device=args.device,
            plan_horizon=args.horizon,
            plan_samples=args.samples,
            plan_iters=args.iters,
        ),
    )
    env_cfg = BallEnvConfig()
    env = BallEnv(env_cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    successes, dists, surprises = 0, [], []
    for ep in range(args.episodes):
        frame = env.reset()
        lo, hi = env_cfg.radius * 3, env_cfg.size - env_cfg.radius * 3
        goal_pos = rng.uniform(lo, hi, size=2)
        goal_frame = make_goal_frame(env, goal_pos)
        eng.reset()

        for t in range(args.steps):
            s = eng.surprise(frame)
            surprises.append(s)
            action = eng.act(frame, goal_frame)
            frame = env.step(action)

        dist = float(np.linalg.norm(env.pos - goal_pos))
        dists.append(dist)
        successes += int(dist < env_cfg.radius * 3)
        if args.verbose:
            print(f"  эпизод {ep}: dist={dist:.2f}")

    print(
        f"успех {successes}/{args.episodes} = {successes / args.episodes:.0%} | "
        f"средняя дистанция {np.mean(dists):.2f}px | "
        f"среднее удивление {np.mean(surprises):.4f}"
    )


def cmd_demo(args: argparse.Namespace) -> None:
    eng = OpenJEPAEngine.from_checkpoint(args.ckpt, RuntimeConfig(device=args.device))
    env = BallEnv(BallEnvConfig(), seed=args.seed)
    frame = env.reset()
    goal_pos = np.array([env.cfg.size * 0.75, env.cfg.size * 0.3])
    goal_frame = make_goal_frame(env, goal_pos)
    eng.reset()

    frames = [frame.copy()]
    t0 = time.time()
    for _ in range(args.steps):
        action = eng.act(frame, goal_frame)
        frame = env.step(action)
        frames.append(frame.copy())
    dt = time.time() - t0

    out = Path(args.out)
    _save_strip(frames, out)
    print(
        f"кадров {len(frames)} | {dt / len(frames) * 1000:.1f} мс/шаг | "
        f"финальная дистанция {np.linalg.norm(env.pos - goal_pos):.2f}px | сохранено {out}"
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Минимальный HTTP-сервер без внешних зависимостей.

    POST /act  {"frame": "<base64 png/raw>", "goal": "<base64>"} -> {"action": [...]}
    GET  /health -> {"ok": true}
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    eng = OpenJEPAEngine.from_checkpoint(args.ckpt, RuntimeConfig(device=args.device))

    def decode(payload: str) -> np.ndarray:
        raw = base64.b64decode(payload)
        try:
            from PIL import Image

            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            size = int(np.sqrt(len(raw) / 3))
            return np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тише в консоли
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/act":
                self._json(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                frame = decode(req["frame"])
                goal = decode(req["goal"])
                action = eng.act(frame, goal)
                self._json(200, {"action": action.tolist()})
            except Exception as exc:  # pragma: no cover - сетевой путь
                self._json(400, {"error": str(exc)})

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"слушаю http://{args.host}:{args.port}  (POST /act, GET /health)")
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(prog="openjepa", description="OpenJEPA runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("demo")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="runs/demo.png")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("serve")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
