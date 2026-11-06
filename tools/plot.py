"""Графики обучения и планирования без matplotlib (только torchvision/PIL).

    python tools/plot.py --run runs/toy --out runs/toy/curves.png

Зачем без matplotlib: лишняя тяжёлая зависимость ради двух картинок.
Здесь SVG строится вручную — он лёгкий, векторный и открывается везде.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_log(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    return cols


def sparkline(values: list[float], width: int = 900, height: int = 180) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = 20 + (width - 40) * i / max(1, n - 1)
        y = height - 20 - (height - 40) * (v - lo) / span
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="#11151c"/>'
        f'<polyline points="{poly}" fill="none" stroke="#5ec8a0" stroke-width="2"/>'
        f'<text x="20" y="16" fill="#8892a0" font-family="monospace" font-size="12">'
        f"min={lo:.4g}  max={hi:.4g}  n={n}</text></svg>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="папка прогона с log.csv")
    ap.add_argument("--out", default=None, help="куда сохранить svg")
    args = ap.parse_args()

    run = Path(args.run)
    log = run / "log.csv"
    if not log.exists():
        raise SystemExit(f"нет файла {log}")

    cols = read_log(log)
    keys = [k for k in ("loss", "pred", "sigreg", "val_pred", "val_std") if k in cols]
    if not keys:
        raise SystemExit("в логе нет ожидаемых колонок")

    parts = []
    for k in keys:
        svg = sparkline(cols[k])
        parts.append(
            f'<div style="margin:8px 0"><div style="color:#c8d0dc;'
            f'font-family:monospace;font-size:13px">{k}</div>{svg}</div>'
        )

    html = (
        "<html><body style='background:#0b0e13;margin:16px'>"
        + "".join(parts)
        + "</body></html>"
    )
    out = Path(args.out) if args.out else run / "curves.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"сохранено: {out}")

    # краткая сводка в консоль
    for k in keys:
        v = cols[k]
        print(f"  {k:10s} first={v[0]:.4f}  last={v[-1]:.4f}  best={min(v):.4f}")


if __name__ == "__main__":
    main()
