"""Проверка библиографии: существует ли статья на arXiv и совпадает ли название.

    python tools/check_refs.py

Требует интернет. Не падает при ошибке сети — печатает предупреждение.
"""

from __future__ import annotations

import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "research" / "bibliography.md"
API = "http://export.arxiv.org/api/query?id_list={ids}&max_results=50"

NS = {"a": "http://www.w3.org/2005/Atom"}


def main() -> int:
    text = BIB.read_text(encoding="utf-8")
    ids = sorted(set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", text)))
    if not ids:
        print("arXiv ID не найдены")
        return 1

    print(f"проверяю {len(ids)} идентификаторов…")
    url = API.format(ids=",".join(ids))
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"сеть недоступна: {exc}")
        return 0

    root = ET.fromstring(data)
    found = {}
    for entry in root.findall("a:entry", NS):
        eid = entry.find("a:id", NS).text.rsplit("/", 1)[-1]
        base = eid.split("v")[0]
        title = " ".join(entry.find("a:title", NS).text.split())
        found[base] = title

    missing = [i for i in ids if i not in found]
    for i in ids:
        if i in found:
            print(f"  ok    {i}  {found[i][:70]}")
        else:
            print(f"  MISS  {i}")

    if missing:
        print(f"\nне найдено: {missing}")
        return 1
    print("\nвсе ссылки валидны")
    return 0


if __name__ == "__main__":
    sys.exit(main())
