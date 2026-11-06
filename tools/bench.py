"""Замер латентности рантайма: энкодер, предиктор, CEM-план.

    python tools/bench.py --ckpt runs/toy/best.pt --iters 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openjepa.planning.cem import CEMConfig, CEMPlanner  # noqa: E402
from openjepa.runtime.engine import OpenJEPAEngine, RuntimeConfig  # noqa: E402


def timeit(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()

    eng = OpenJEPAEngine.from_checkpoint(
        args.ckpt, RuntimeConfig(plan_horizon=args.horizon, plan_samples=args.samples)
    )
    dev = eng.device
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    goal = np.zeros((64, 64, 3), dtype=np.uint8)

    with torch.no_grad():
        z = eng.encode(frame)
        zg = eng.encode(goal)
        a = torch.zeros(1, 2, device=dev)

        t_enc = timeit(lambda: eng.encode(frame), args.iters)
        t_pred = timeit(lambda: eng.model.predict_next(z, a), args.iters)
        t_plan = timeit(lambda: eng.planner.plan(z, zg), max(3, args.iters // 5))

    params = sum(p.numel() for p in eng.model.parameters())
    print(f"device        {dev}")
    print(f"parameters    {params:,}")
    print(f"encode        {t_enc:7.2f} ms")
    print(f"predict 1step {t_pred:7.2f} ms")
    print(f"CEM plan      {t_plan:7.2f} ms   (H={args.horizon}, S={args.samples})")
    print(f"control loop  {1000.0 / (t_enc + t_plan):7.1f} Hz  (без батчинга)")


if __name__ == "__main__":
    main()
