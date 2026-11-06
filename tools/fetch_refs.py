"""Скачивает PDF статей с arXiv в refs/ с проверкой, что это действительно PDF.

    python tools/fetch_refs.py 2511.08544 2610.39302
    python tools/fetch_refs.py --from-bib research/bibliography.md

Зачем проверка: arXiv при перегрузке или при неверном ID отдаёт HTML-страницу
с ошибкой и кодом 200. Если сохранить её как `.pdf`, потом «файл не
открывается» и час на выяснение, что дело не в ридере.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "refs"
UA = "openjepa-refs/0.1 (+research; contact: local)"


def short_name(arxiv_id: str) -> str:
    """Короткое имя файла: только ID, остальное добавим вручную при желании."""
    return arxiv_id.replace("/", "_")


def fetch(arxiv_id: str, out_dir: Path, force: bool = False) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{short_name(arxiv_id)}.pdf"
    if path.exists() and not force:
        print(f"  skip  {path.name} (уже есть)")
        return True

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"  FAIL  {arxiv_id}: {exc}")
        return False

    if not data.startswith(b"%PDF"):
        head = data[:80].decode("utf-8", "replace").replace("\n", " ")
        print(f"  FAIL  {arxiv_id}: не PDF (начало: {head!r})")
        return False

    path.write_bytes(data)
    print(f"  ok    {path.name}  {len(data) / 1024:.0f} KiB")
    return True


def ids_from_bib(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", text)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="arXiv ID")
    ap.add_argument("--from-bib", default=None, help="взять все ID из библиографии")
    ap.add_argument("--out", default=str(REFS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids = list(args.ids)
    if args.from_bib:
        ids += ids_from_bib(Path(args.from_bib))

    if not ids:
        print("нечего скачивать: укажите arXiv ID или --from-bib")
        return 2

    out = Path(args.out)
    print(f"скачиваю {len(ids)} статей в {out}")
    ok = sum(fetch(i, out, args.force) for i in ids)
    print(f"\n{ok}/{len(ids)} успешно")
    return 0 if ok == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
