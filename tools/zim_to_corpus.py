"""Выгружает статьи из ZIM-архива в JSONL с чистым текстом.

    python tools/zim_to_corpus.py <zim> --out data/wiki.jsonl --limit 5000
    python tools/zim_to_corpus.py <zim> --out data/wiki.jsonl --namespace C --min-chars 300

Зачем. ZIM — это архив с HTML, CSS, картинками и скриптами. Для обучения
языковой части нужен только текст. Скрипт проходит по статьям, вырезает
текст, отбрасывает служебные namespace'ы и слишком короткие огрызки, и
пишет по одному JSON-объекту на строку:

    {"title": "...", "url": "...", "text": "..."}

Почему JSONL, а не один большой JSON: его можно читать потоково и
останавливать на любом месте, не парся гигабайты.

Namespace'ы ZIM:
    C — статьи (то, что нужно)
    M — метаданные (заголовок архива, описание)
    W — полнотекстовый индекс и служебное
    X — поисковый индекс (xapian)
    A/I/- — картинки и прочее
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zim_reader import ZimReader, ZimError, strip_html  # noqa: E402

# Служебные блоки MediaWiki, которые не являются содержимым статьи.
NOISE = re.compile(
    r"(?is)(редактировать|править код|править вики-текст)\s*"
    r"|\[\s*править\s*\|?\s*править код\s*\]"
    r"|Материал из Википедии"
)


def clean_text(text: str, min_chars: int) -> str:
    text = NOISE.sub(" ", text)
    # Ссылки-сноски вида [1], [12]
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) >= min_chars else ""


def main() -> int:
    ap = argparse.ArgumentParser(description="ZIM -> JSONL с текстом")
    ap.add_argument("zim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--namespace", default="C")
    ap.add_argument("--limit", type=int, default=0, help="0 = все статьи")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--skip-redirects", action="store_true", default=True)
    ap.add_argument("--progress", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    t0 = time.time()

    with ZimReader(args.zim) as z, out.open("w", encoding="utf-8") as f:
        total = z.article_count if not args.limit else min(args.limit, z.article_count)
        for i in range(total):
            try:
                d = z.dirent(i)
            except ZimError:
                skipped += 1
                continue
            if d.namespace != args.namespace:
                continue
            if args.skip_redirects and d.is_redirect:
                continue
            if z.mimetype_of(i) not in ("text/html", "text/plain", "text/html; charset=iso-8859-1"):
                continue

            try:
                body = z.content(i)
            except ZimError:
                skipped += 1
                continue

            text = clean_text(strip_html(body), args.min_chars)
            if not text or len(text) > args.max_chars:
                skipped += 1
                continue

            f.write(
                json.dumps(
                    {"title": d.title, "url": d.url, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if args.progress and written % args.progress == 0:
                rate = written / max(1e-9, time.time() - t0)
                print(f"  {written} статей, {rate:.0f}/с, {out.stat().st_size/1e6:.1f} MB")

    dt = time.time() - t0
    size_mb = out.stat().st_size / 1e6 if out.exists() else 0
    print(f"\nготово: {written} статей, пропущено {skipped}, {size_mb:.1f} MB, {dt:.0f} с")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
