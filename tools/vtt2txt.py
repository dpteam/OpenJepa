"""Convert YouTube auto-generated WebVTT subtitles into clean, deduplicated plain text.

Auto-subs repeat rolling lines; this collapses them into one flowing paragraph per
sentence chunk while keeping a timestamped variant for reference.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+-->\s+")
TAG = re.compile(r"<[^>]+>")


def parse(vtt_path: Path):
    lines = vtt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cues: list[tuple[float, str]] = []
    cur_start = None
    buf: list[str] = []
    for line in lines:
        m = TS.match(line)
        if m:
            if cur_start is not None and buf:
                cues.append((cur_start, " ".join(buf)))
            h, mi, s, ms = (int(x) for x in m.groups())
            cur_start = h * 3600 + mi * 60 + s + ms / 1000
            buf = []
            continue
        if line.strip() == "":
            if cur_start is not None and buf:
                cues.append((cur_start, " ".join(buf)))
                cur_start, buf = None, []
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        text = TAG.sub("", line).strip()
        if text:
            buf.append(text)
    if cur_start is not None and buf:
        cues.append((cur_start, " ".join(buf)))
    return cues


def dedupe(cues):
    """Rolling captions: each cue often repeats the tail of the previous one."""
    out: list[tuple[float, str]] = []
    prev_words: list[str] = []
    for start, text in cues:
        words = text.split()
        # find longest overlap between tail of prev and head of current
        overlap = 0
        for k in range(min(len(prev_words), len(words)), 0, -1):
            if prev_words[-k:] == words[:k]:
                overlap = k
                break
        new = words[overlap:]
        if new:
            out.append((start, " ".join(new)))
        prev_words = words
    return out


def fmt(ts: float) -> str:
    ts = int(ts)
    return f"{ts // 60:02d}:{ts % 60:02d}"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: vtt2txt.py <in.vtt> <out.txt>")
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cues = dedupe(parse(src))
    dst.parent.mkdir(parents=True, exist_ok=True)

    # flowing text
    flow = " ".join(t for _, t in cues)
    flow = re.sub(r"\s+", " ", flow).strip()

    # timestamped blocks, ~45s each
    blocks: list[str] = []
    bucket_start, bucket = None, []
    for start, text in cues:
        if bucket_start is None:
            bucket_start = start
        bucket.append(text)
        if start - bucket_start >= 45:
            blocks.append(f"[{fmt(bucket_start)}] {' '.join(bucket)}")
            bucket_start, bucket = None, []
    if bucket:
        blocks.append(f"[{fmt(bucket_start)}] {' '.join(bucket)}")

    dst.write_text(
        "# Транскрипт\n\n## Сплошной текст\n\n"
        + flow
        + "\n\n## С таймкодами\n\n"
        + "\n\n".join(blocks)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {dst} ({len(flow)} chars, {len(blocks)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
