"""Собирает отчёт по прогону в markdown: метрики, графики, проверки.

    python tools/report.py --run runs/toy

Пишет `runs/<id>/report.md`. Отчёт самодостаточен: числа, диагноз, что
делать дальше. Это то, что должно прикладываться к любому результату
(см. Definition of Done в ROADMAP.md).
"""

from __future__ import annotations

import argparse
import csv
import json
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


def diagnose(cols: dict[str, list[float]]) -> list[tuple[str, str]]:
    """Возвращает список (статус, сообщение). Статус: ok / warn / fail."""
    out: list[tuple[str, str]] = []
    if not cols:
        return [("fail", "лог пуст")]

    pred = cols.get("pred", [])
    reg = cols.get("sigreg", [])
    vstd = cols.get("val_std", [])
    vpred = cols.get("val_pred", [])

    if reg:
        if reg[-1] < 0.5:
            out.append(("fail", f"SIGReg → {reg[-1]:.3f}: вероятен коллапс латента"))
        elif reg[-1] > 100:
            out.append(("warn", f"SIGReg = {reg[-1]:.1f}: слишком большой, λ велик?"))
        else:
            out.append(("ok", f"SIGReg в рабочем коридоре ({reg[-1]:.2f})"))

    if vstd:
        if vstd[-1] < 0.05:
            out.append(("fail", f"val_std = {vstd[-1]:.4f}: латент схлопнулся в eval"))
        else:
            out.append(("ok", f"val_std = {vstd[-1]:.3f}: латент не вырожден"))

    if pred and len(pred) > 2:
        drop = (pred[0] - pred[-1]) / max(pred[0], 1e-9)
        if drop < 0.2:
            out.append(("warn", f"pred_loss упал всего на {drop:.0%}"))
        else:
            out.append(("ok", f"pred_loss упал на {drop:.0%} ({pred[0]:.3f} → {pred[-1]:.3f})"))

    if vpred and pred:
        ratio = vpred[-1] / max(pred[-1], 1e-9)
        if ratio > 3:
            out.append(("warn", f"val_pred/pred = {ratio:.1f}×: проверьте BN-статистики"))
        else:
            out.append(("ok", f"val_pred/pred = {ratio:.2f}×: train и eval согласованы"))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    log = run / "log.csv"
    if not log.exists():
        raise SystemExit(f"нет {log}")

    cols = read_log(log)
    cfg_path = run / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    lines = [f"# Отчёт: `{run.name}`", ""]
    if cfg:
        m = cfg.get("model", {})
        t = cfg.get("train", {})
        lines += [
            "## Конфигурация",
            "",
            f"* модель: `dim={m.get('dim')}`, `depth={m.get('depth')}`, "
            f"`proj_dim={m.get('proj_dim')}`, `ctx={m.get('ctx', 1)}`",
            f"* обучение: `lr={t.get('lr')}`, `batch={t.get('batch_size')}`, "
            f"`epochs={t.get('epochs')}`, `λ={m.get('lamb')}`",
            f"* иерархия: {m.get('hierarchical', False)}",
            "",
        ]

    lines += ["## Динамика", "", "| метрика | старт | конец | лучший |", "|---|---|---|---|"]
    for k in ("loss", "pred", "sigreg", "val_pred", "val_std"):
        v = cols.get(k)
        if not v:
            continue
        lines.append(f"| `{k}` | {v[0]:.4f} | {v[-1]:.4f} | {min(v):.4f} |")
    lines.append("")

    lines += ["## Диагноз", ""]
    for status, msg in diagnose(cols):
        icon = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[status]
        lines.append(f"* **{icon}** — {msg}")
    lines.append("")

    lines += [
        "## Что дальше",
        "",
        "Смотрите `ROADMAP.md`, текущая фаза. Минимальный набор для честного",
        "результата: 3 seed'а, baseline, тест на сдвиг цвета (`docs/10-eval.md`).",
        "",
    ]

    out = run / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"сохранено: {out}")
    for status, msg in diagnose(cols):
        print(f"  [{status}] {msg}")


if __name__ == "__main__":
    main()
