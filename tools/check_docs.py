"""Проверка документации: битые внутренние ссылки и относительные пути.

    python tools/check_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE = re.compile(r"`([^`]+\.(?:md|py|yaml|yml|txt|svg))`")


def iter_md() -> list[Path]:
    return [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]


def main() -> int:
    problems: list[str] = []
    files = iter_md()

    for path in files:
        text = path.read_text(encoding="utf-8")
        for _, target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                problems.append(f"{path.relative_to(ROOT)} -> {target} (нет файла)")

        # проверяем упоминания файлов в обратных кавычках
        for name in CODE.findall(text):
            if any(c in name for c in "*?<>") or " " in name:
                continue
            candidates = [
                ROOT / name,
                path.parent / name,
                ROOT / "src" / "openjepa" / name,   # ссылки вида `models/vit.py`
                ROOT / "docs" / name,               # ссылки вида `05-obuchenie.md`
            ]
            if not any(c.exists() for c in candidates):
                problems.append(f"{path.relative_to(ROOT)} -> `{name}` (нет файла)")

    print(f"проверено файлов: {len(files)}")
    if problems:
        print(f"проблем: {len(problems)}")
        for p in problems:
            print("  " + p)
        return 1
    print("все ссылки валидны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
