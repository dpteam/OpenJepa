"""Тесты ZIM-читателя.

Тест пропускается, если локальных архивов нет: они не входят в репозиторий.
Путь можно задать переменной окружения `OPENJEPA_ZIM`.

    set OPENJEPA_ZIM=H:\\...\\gutenberg_ru_all_2026-01.zim
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from zim_reader import MAGIC, ZimReader, ZimError  # noqa: E402

DEFAULT = Path(r"H:\pub_h\Mirrors\WikiS\Kiwix\ZIMs\gutenberg_ru_all_2026-01.zim")


def _zim() -> Path | None:
    p = Path(os.environ.get("OPENJEPA_ZIM", DEFAULT))
    return p if p.exists() else None


def test_magic_constant():
    assert MAGIC == 72173914


def test_rejects_non_zim():
    import tempfile

    bad = Path(tempfile.mkdtemp(prefix="openjepa_zim_")) / "bad.zim"
    bad.write_bytes(b"not a zim file at all" * 10)
    try:
        ZimReader(bad)
    except ZimError:
        return
    raise AssertionError("должен был отвергнуть не-ZIM файл")


def test_rejects_missing_file():
    try:
        ZimReader("nonexistent_12345.zim")
    except ZimError:
        return
    raise AssertionError("должен был сообщить об отсутствии файла")


def test_header_and_mimetypes():
    p = _zim()
    if p is None:
        print("    (пропущено: нет локального ZIM)")
        return
    with ZimReader(p) as z:
        assert z.major >= 5
        assert z.article_count > 0
        assert z.cluster_count > 0
        assert len(z.mimetypes) > 0
        assert z.url_ptr_pos < z.size


def test_dirents_and_content():
    p = _zim()
    if p is None:
        print("    (пропущено: нет локального ZIM)")
        return
    with ZimReader(p) as z:
        n = min(50, z.article_count)
        seen = 0
        for i in range(n):
            d = z.dirent(i)
            assert d.url != "" or d.is_redirect
            if d.is_redirect:
                assert d.redirect_index < z.article_count
                continue
            body = z.content(i)
            assert isinstance(body, bytes)
            seen += 1
        assert seen > 0


def test_url_search():
    p = _zim()
    if p is None:
        print("    (пропущено: нет локального ZIM)")
        return
    with ZimReader(p) as z:
        d0 = z.dirent(0)
        # ищем по URL первой записи — должен найтись тот же индекс
        assert z.find(d0.url, namespace=d0.namespace) is not None


def test_offsets_are_monotonic():
    p = _zim()
    if p is None:
        print("    (пропущено: нет локального ZIM)")
        return
    with ZimReader(p) as z:
        prev = -1
        for i in range(0, min(1000, z.article_count)):
            off = z.url_ptrs[i]
            assert off > prev, f"url_ptrs не возрастает на {i}"
            prev = off


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
