"""Замер одного чекпоинта по всем метрикам, без переобучения.

    python tools/measure.py --ckpt runs/sweep3/proj8/best.pt

Полезно, когда меняется методика замера: переобучать не нужно.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sweep import measure  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    res = measure(Path(args.ckpt))
    print(f"чекпоинт: {args.ckpt}")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:8s} {v:.4f}")
        else:
            print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
