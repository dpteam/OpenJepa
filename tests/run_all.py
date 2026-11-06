"""Запуск всех тестов без pytest:  python tests/run_all.py"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    failed = 0
    total = 0
    for path in sorted(ROOT.glob("test_*.py")):
        mod = load(path)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                print(f"ok    {path.stem}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
