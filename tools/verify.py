"""Проверка всего пайплайна одной командой.

    python tools/verify.py

Что делает:
  1. гоняет тесты;
  2. обучает smoke-конфиг;
  3. проверяет, что латент не схлопнулся и train/eval согласованы;
  4. запускает eval и проверяет, что планировщик вообще двигает объект.

Возвращает 0, если всё в порядке.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def step(name: str) -> None:
    print(f"\n=== {name} ===")


def main() -> int:
    env_extra = {"PYTHONPATH": str(ROOT / "src")}
    import os

    env = {**os.environ, **env_extra}
    failures: list[str] = []

    step("1/4 тесты")
    r = run([PY, "tests/run_all.py"], env=env)
    tail = (r.stdout or "").strip().splitlines()[-1:] or ["нет вывода"]
    print(tail[0])
    if r.returncode != 0:
        failures.append("тесты")

    out = Path(tempfile.mkdtemp(prefix="openjepa_verify_"))

    step("2/4 smoke-обучение")
    r = run(
        [PY, "-m", "openjepa.runtime.cli", "train", "--config", "configs/smoke.yaml",
         "--out", str(out)],
        env=env,
    )
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        failures.append("обучение")
        return _report(failures)
    print((r.stdout or "").strip().splitlines()[-1])

    step("3/4 проверка латента и согласованности train/eval")
    check = f"""
import sys, torch
sys.path.insert(0, {str(ROOT / 'src')!r})
from openjepa.runtime import OpenJEPAEngine
from openjepa.data.synthetic import TrajectoryDataset, collate
eng = OpenJEPAEngine.from_checkpoint({str(out / 'best.pt')!r})
m = eng.model
ds = TrajectoryDataset(n_traj=8, length=8, seed=991)
b = collate([ds[i] for i in range(8)])
fr, ac = b['frames'].to(eng.device), b['actions'].to(eng.device)
res = {{}}
for mode in ('train', 'eval'):
    getattr(m, mode)()
    with torch.no_grad():
        o = m(fr, ac)
        z = m.encode_sequence(fr).float().reshape(-1, m.cfg.proj_dim)
    res[mode] = dict(pred=float(o['pred_loss']), std=float(z.std(0).min()),
                     norm=float(z.norm(dim=-1).mean()))
print(res)
ok = res['eval']['std'] > 0.05 and res['train']['pred'] < 0.5
ratio = max(res['eval']['pred'], 1e-9) / max(res['train']['pred'], 1e-9)
ok = ok and ratio < 3.0
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
"""
    r = run([PY, "-c", check], env=env)
    print((r.stdout or "").strip())
    if r.returncode != 0:
        failures.append("качество латента")

    step("4/4 планирование")
    r = run(
        [PY, "-m", "openjepa.runtime.cli", "eval", "--ckpt", str(out / "best.pt"),
         "--episodes", "5", "--steps", "25", "--samples", "128"],
        env=env,
    )
    print((r.stdout or "").strip() or (r.stderr or "").strip()[-500:])
    if r.returncode != 0:
        failures.append("планирование")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print("\n" + "=" * 50)
    if failures:
        print("ПРОВАЛ: " + ", ".join(failures))
        return 1
    print("ВСЁ В ПОРЯДКЕ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
