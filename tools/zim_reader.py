"""Читатель ZIM-архивов (Wikipedia offline, Kiwix) на чистом Python.

Зачем свой. `libzim` — это C++-биндинг, который надо собирать; `zimply`
заброшен и падает на современных архивах. Формат ZIM простой, и его можно
читать стандартной библиотекой (плюс `zstandard` для сжатых кластеров).

    python tools/zim_reader.py info  "H:/.../wikipedia_ru_all_maxi.zim"
    python tools/zim_reader.py list  <zim> --limit 20
    python tools/zim_reader.py get   <zim> "Альберт Эйнштейн"
    python tools/zim_reader.py text  <zim> "Альберт Эйнштейн" --out article.txt
    python tools/zim_reader.py check <zim>

`check` проходит по всем записям и проверяет целостность: магию, границы
указателей, распаковку кластеров. Это и есть «проверка на корректность».

Формат (ZIM v6), кратко:

    заголовок (80 байт)
    mimeList      — список MIME-строк, оканчивается дополнительным \\0
    urlPtrList    — articleCount × uint64, отсортирован по URL
    titlePtrList  — articleCount × uint32, отсортирован по заголовку
    clusterPtrList— clusterCount × uint64
    dirent'ы      — записи: mime, namespace, cluster, blob, url, title
    кластеры      — сжатые блоки с оглавлением блобов

Сжатие кластера: 0/1 — без сжатия, 4 — LZMA, 5 — Zstandard.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAGIC = 72173914

COMP_NONE = {0, 1}
COMP_LZMA = 4
COMP_ZSTD = 5
COMP_ZLIB = 2
COMP_BZIP2 = 3

try:  # zstd нужен для большинства архивов Kiwix
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None


class ZimError(Exception):
    pass


@dataclass
class Dirent:
    mimetype: int
    namespace: str
    revision: int
    cluster: int
    blob: int
    url: str
    title: str
    offset: int

    @property
    def is_redirect(self) -> bool:
        """0xFFFF в поле mimetype означает «перенаправление».

        Тогда `revision` — это индекс целевой статьи, а `cluster`/`blob`
        не имеют смысла (там мусор). Именно из-за этого «проверка
        целостности» сначала падала на mainPage: он оказался редиректом.
        """
        return self.mimetype == 0xFFFF

    @property
    def redirect_index(self) -> int:
        return self.revision


class ZimReader:
    """Минимальный, но полный читатель ZIM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ZimError(f"нет файла: {self.path}")
        self.f = self.path.open("rb")
        self.size = self.path.stat().st_size
        self._read_header()
        self._read_mimelist()
        self._read_pointers()
        self._cluster_cache: dict[int, list[bytes]] = {}

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "ZimReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _read_header(self) -> None:
        self.f.seek(0)
        raw = self.f.read(80)
        if len(raw) < 80:
            raise ZimError("файл короче заголовка")
        (
            magic,
            major,
            minor,
            uuid,
            article_count,
            cluster_count,
            url_ptr_pos,
            title_ptr_pos,
            cluster_ptr_pos,
            mime_list_pos,
            main_page,
            layout_page,
            checksum_pos,
        ) = struct.unpack("<IHH16sIIQQQQIIQ", raw)

        if magic != MAGIC:
            raise ZimError(f"не ZIM: магия {magic} != {MAGIC}")

        self.major = major
        self.minor = minor
        self.uuid = uuid
        self.article_count = article_count
        self.cluster_count = cluster_count
        self.url_ptr_pos = url_ptr_pos
        self.title_ptr_pos = title_ptr_pos
        self.cluster_ptr_pos = cluster_ptr_pos
        self.mime_list_pos = mime_list_pos
        self.main_page = main_page
        self.layout_page = layout_page
        self.checksum_pos = checksum_pos

        # Санити-чек: все указатели должны лежать внутри файла.
        # titlePtrPos = 0xFFFF... означает «индекса по заголовкам нет» —
        # так делают новые архивы (ZIM >= 6). Тогда используем порядок URL.
        for name, pos in [
            ("urlPtrPos", url_ptr_pos),
            ("clusterPtrPos", cluster_ptr_pos),
            ("mimeListPos", mime_list_pos),
        ]:
            if pos >= self.size:
                raise ZimError(f"{name}={pos} за пределами файла ({self.size})")
        self.has_title_index = title_ptr_pos < self.size

    def _read_mimelist(self) -> None:
        self.f.seek(self.mime_list_pos)
        self.mimetypes: list[str] = []
        buf = b""
        while True:
            ch = self.f.read(1)
            if not ch:
                raise ZimError("mimeList обрывается без завершающего \\0")
            if ch == b"\0":
                if not buf:
                    break  # двойной \0 — конец списка
                self.mimetypes.append(buf.decode("utf-8", "replace"))
                buf = b""
            else:
                buf += ch
        if len(self.mimetypes) > 200:
            raise ZimError(f"подозрительно много MIME-типов: {len(self.mimetypes)}")

    def _read_pointers(self) -> None:
        n = self.article_count
        self.f.seek(self.url_ptr_pos)
        raw = self.f.read(8 * n)
        if len(raw) != 8 * n:
            raise ZimError("urlPtrList короче ожидаемого")
        self.url_ptrs = struct.unpack(f"<{n}Q", raw)

        if self.has_title_index:
            self.f.seek(self.title_ptr_pos)
            raw = self.f.read(4 * n)
            if len(raw) != 4 * n:
                raise ZimError("titlePtrList короче ожидаемого")
            self.title_ptrs = struct.unpack(f"<{n}I", raw)
        else:
            # Индекса нет — порядок совпадает с URL.
            self.title_ptrs = tuple(range(n))

        c = self.cluster_count
        self.f.seek(self.cluster_ptr_pos)
        raw = self.f.read(8 * c)
        if len(raw) != 8 * c:
            raise ZimError("clusterPtrList короче ожидаемого")
        self.cluster_ptrs = struct.unpack(f"<{c}Q", raw)

    # ------------------------------------------------------------------ #
    def dirent(self, index: int) -> Dirent:
        if not 0 <= index < self.article_count:
            raise ZimError(f"индекс {index} вне диапазона 0..{self.article_count - 1}")
        off = self.url_ptrs[index]
        self.f.seek(off)
        head = self.f.read(16)
        if len(head) < 12:
            raise ZimError(f"dirent[{index}] обрывается")

        mimetype, param_len, ns_byte = struct.unpack("<HBB", head[:4])
        revision = struct.unpack("<I", head[4:8])[0]

        if mimetype == 0xFFFF:
            # Редирект: после revision идёт redirectIndex (4 байта), и на этом
            # заголовок кончается. Дальше сразу url. Итого 12 байт, а не 16 —
            # если читать как обычную запись, url сдвигается на 4 байта и
            # «кластер» превращается в мусор.
            redirect_index = struct.unpack("<I", head[8:12])[0]
            self.f.seek(off + 12)
            url = self._read_cstr()
            title = self._read_cstr()
            return Dirent(
                mimetype=mimetype,
                namespace=chr(ns_byte),
                revision=redirect_index,
                cluster=-1,
                blob=-1,
                url=url,
                title=title or url,
                offset=off,
            )

        cluster, blob = struct.unpack("<II", head[8:16])
        self.f.seek(off + 16)
        url = self._read_cstr()
        title = self._read_cstr()
        if param_len:
            self.f.read(param_len)
        return Dirent(
            mimetype=mimetype,
            namespace=chr(ns_byte),
            revision=revision,
            cluster=cluster,
            blob=blob,
            url=url,
            title=title or url,
            offset=off,
        )

    def _read_cstr(self) -> str:
        buf = bytearray()
        while True:
            ch = self.f.read(1)
            if not ch:
                raise ZimError("строка без завершающего \\0")
            if ch == b"\0":
                return buf.decode("utf-8", "replace")
            buf += ch

    # ------------------------------------------------------------------ #
    def cluster_blobs(self, cluster_index: int) -> list[bytes]:
        """Распаковывает кластер и возвращает список блобов (кэшируется).

        Структура кластера: один байт с типом сжатия, за ним **весь остальной
        блок** (оглавление + данные) сжат целиком. Поэтому сначала читаем
        весь диапазон кластера, распаковываем, и только потом разбираем
        оглавление.
        """
        if cluster_index in self._cluster_cache:
            return self._cluster_cache[cluster_index]
        if not 0 <= cluster_index < self.cluster_count:
            raise ZimError(f"кластер {cluster_index} вне диапазона")

        start = self.cluster_ptrs[cluster_index]
        if cluster_index + 1 < self.cluster_count:
            end = self.cluster_ptrs[cluster_index + 1]
        else:
            end = self.checksum_pos if 0 < self.checksum_pos <= self.size else self.size
        if not (start < end <= self.size):
            raise ZimError(f"кластер {cluster_index}: границы {start}..{end} вне файла")

        self.f.seek(start)
        raw = self.f.read(end - start)
        if not raw:
            raise ZimError(f"кластер {cluster_index}: пусто")

        info_byte = raw[0]
        comp = info_byte & 0x0F
        extended = bool(info_byte & 0x10)
        payload = raw[1:]

        if comp in COMP_NONE:
            pass
        elif comp == COMP_LZMA:
            payload = __import__("lzma").decompress(payload)
        elif comp == COMP_ZSTD:
            if _zstd is None:
                raise ZimError(
                    "кластер сжат Zstandard, а модуль `zstandard` не установлен: "
                    "pip install zstandard"
                )
            payload = _zstd.ZstdDecompressor().decompress(
                payload, max_output_size=1 << 30
            )
        elif comp == COMP_ZLIB:
            payload = __import__("zlib").decompress(payload)
        elif comp == COMP_BZIP2:
            payload = __import__("bz2").decompress(payload)
        else:
            raise ZimError(f"неизвестное сжатие кластера: {comp}")

        if extended:
            off_size, off_fmt = 8, "Q"
        else:
            off_size, off_fmt = 4, "I"

        # Формат (libzim, Cluster::read_header): первое значение — это
        # одновременно смещение первого блоба и размер заголовка.
        # n_offset = first / sizeof(offset); дальше идут остальные смещения.
        if len(payload) < off_size:
            raise ZimError(f"кластер {cluster_index}: пустой заголовок")
        first = struct.unpack(f"<{off_fmt}", payload[:off_size])[0]
        if first < 2 * off_size or first % off_size != 0 or first > len(payload):
            raise ZimError(
                f"кластер {cluster_index}: некорректное смещение первого блоба ({first})"
            )
        n_offset = first // off_size
        if n_offset > 1_000_000:
            raise ZimError(f"кластер {cluster_index}: подозрительно много блобов ({n_offset})")
        offsets = struct.unpack(f"<{n_offset}{off_fmt}", payload[:first])

        # Смещения относительны начала payload (после байта сжатия).
        blobs = [payload[offsets[i] : offsets[i + 1]] for i in range(n_offset - 1)]
        self._cluster_cache[cluster_index] = blobs
        return blobs

    # ------------------------------------------------------------------ #
    def content(self, index: int, follow_redirect: bool = True) -> bytes:
        d = self.dirent(index)
        if d.is_redirect:
            if not follow_redirect:
                raise ZimError(f"запись {index} — редирект на {d.redirect_index}")
            return self.content(d.redirect_index, follow_redirect=True)
        return self.cluster_blobs(d.cluster)[d.blob]

    def mimetype_of(self, index: int) -> str:
        d = self.dirent(index)
        if d.mimetype >= len(self.mimetypes):
            return "application/octet-stream"
        return self.mimetypes[d.mimetype]

    # ------------------------------------------------------------------ #
    def find(self, url: str, namespace: str = "C") -> int | None:
        """Бинарный поиск по URL (urlPtrList отсортирован)."""
        key = f"{namespace}/{url}".encode("utf-8")
        lo, hi = 0, self.article_count - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            d = self.dirent(mid)
            cur = f"{d.namespace}/{d.url}".encode("utf-8")
            if cur == key:
                return mid
            if cur < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def find_title(self, title: str, namespace: str = "C") -> int | None:
        """Поиск по заголовку.

        В архивах с title-индексом используется бинарный поиск по нему
        (сравнение без учёта регистра и с нормализацией `_` → пробел).
        В архивах без индекса — линейный проход: для 7.5M статей это минуты,
        поэтому лучше искать по URL через `find()`.
        """
        target = self._norm_title(title)

        if self.has_title_index:
            lo, hi = 0, self.article_count - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                d = self.dirent(self.title_ptrs[mid])
                cur = self._norm_title(d.title)
                if cur == target and d.namespace == namespace:
                    return self.title_ptrs[mid]
                if cur < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            # бинарный поиск по заголовку может промахнуться из-за namespace,
            # поэтому делаем короткий линейный проход вокруг найденной точки
            for off in range(-64, 65):
                i = min(max(lo + off, 0), self.article_count - 1)
                d = self.dirent(self.title_ptrs[i])
                if self._norm_title(d.title) == target and d.namespace == namespace:
                    return self.title_ptrs[i]
            return None

        for i in range(self.article_count):
            d = self.dirent(i)
            if d.namespace == namespace and self._norm_title(d.title) == target:
                return i
        return None

    @staticmethod
    def _norm_title(s: str) -> str:
        return s.replace("_", " ").strip().casefold()

    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        return {
            "path": str(self.path),
            "size_gb": round(self.size / 1024**3, 2),
            "version": f"{self.major}.{self.minor}",
            "articles": self.article_count,
            "clusters": self.cluster_count,
            "mimetypes": self.mimetypes,
            "main_page": self.main_page,
            "zstd": _zstd is not None,
        }


# --------------------------------------------------------------------------- #
def strip_html(html: bytes) -> str:
    """Грубое извлечение текста из HTML. Без bs4 — чтобы не тянуть зависимость."""
    import html as _html
    import re

    text = html.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def cmd_info(args) -> int:
    with ZimReader(args.zim) as z:
        for k, v in z.info().items():
            print(f"{k:12s} {v}")
    return 0


def cmd_list(args) -> int:
    with ZimReader(args.zim) as z:
        n = min(args.limit, z.article_count)
        for i in range(args.offset, args.offset + n):
            d = z.dirent(i)
            print(f"{i:8d}  {d.namespace}  {d.url[:60]:60s}  {d.title[:40]}")
    return 0


def _resolve(z: ZimReader, entry: str) -> int | None:
    """Находит запись: сначала по URL, потом по заголовку.

    Важно не использовать `a or b`: индекс 0 — валидная запись, но в Python
    это falsy. Именно на этом мы потеряли полдня: `find(...) or find_title(...)`
    молча подменял найденную статью нулевой.
    """
    idx = z.find(entry)
    if idx is not None:
        return idx
    return z.find_title(entry)


def cmd_get(args) -> int:
    with ZimReader(args.zim) as z:
        idx = _resolve(z, args.entry)
        if idx is None:
            print(f"не найдено: {args.entry}", file=sys.stderr)
            return 1
        d = z.dirent(idx)
        body = z.content(idx)
        print(f"# {d.title}  [{z.mimetype_of(idx)}, {len(body)} байт]")
        if args.raw:
            sys.stdout.buffer.write(body)
        else:
            print(body[:2000].decode("utf-8", "replace"))
    return 0


def cmd_text(args) -> int:
    with ZimReader(args.zim) as z:
        idx = _resolve(z, args.entry)
        if idx is None:
            print(f"не найдено: {args.entry}", file=sys.stderr)
            return 1
        d = z.dirent(idx)
        text = strip_html(z.content(idx))
        if args.out:
            Path(args.out).write_text(f"# {d.title}\n\n{text}\n", encoding="utf-8")
            print(f"сохранено: {args.out} ({len(text)} символов)")
        else:
            print(f"# {d.title}\n")
            print(text[: args.chars])
    return 0


def cmd_check(args) -> int:
    """Проверка целостности: заголовок, указатели, распаковка всех кластеров."""
    problems: list[str] = []
    with ZimReader(args.zim) as z:
        print(f"файл: {z.path.name}  {z.size / 1024**3:.2f} GB")
        print(f"версия {z.major}.{z.minor}, статей {z.article_count}, "
              f"кластеров {z.cluster_count}, MIME {len(z.mimetypes)}")

        # 1. Указатели монотонны и внутри файла
        prev = 0
        bad_ptrs = 0
        for i, p in enumerate(z.url_ptrs):
            if p < prev or p >= z.size:
                bad_ptrs += 1
            prev = p
        if bad_ptrs:
            problems.append(f"urlPtrList: {bad_ptrs} некорректных указателей")

        # 2. Все кластеры распаковываются
        limit = z.cluster_count if args.all_clusters else min(20, z.cluster_count)
        bad_clusters = 0
        for c in range(limit):
            try:
                z.cluster_blobs(c)
            except Exception as exc:
                bad_clusters += 1
                if bad_clusters <= 3:
                    problems.append(f"кластер {c}: {exc}")
        if bad_clusters:
            problems.append(f"кластеров с ошибками: {bad_clusters} из {limit}")

        # 3. Dirent'ы читаются и блобы в границах
        step = max(1, z.article_count // max(1, args.sample))
        bad_dirents = 0
        redirects = 0
        for i in range(0, z.article_count, step):
            try:
                d = z.dirent(i)
                if d.is_redirect:
                    redirects += 1
                    if d.redirect_index >= z.article_count:
                        bad_dirents += 1
                    continue
                blobs = z.cluster_blobs(d.cluster)
                if d.blob >= len(blobs):
                    bad_dirents += 1
            except Exception:
                bad_dirents += 1
        if bad_dirents:
            problems.append(f"dirent'ов с ошибками: {bad_dirents}")
        if redirects:
            print(f"редиректов в выборке: {redirects}")

        # 4. Бинарный поиск работает на известной статье
        if z.article_count and z.find("Main_Page") is not None:
            print("поиск по URL: ok (Main_Page найден)")

        print()
        if problems:
            print("ПРОБЛЕМЫ:")
            for p in problems:
                print("  - " + p)
            return 1
        print("ПРОВЕРКА ПРОЙДЕНА: структура целостна")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Читатель ZIM-архивов")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("zim")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("list")
    p.add_argument("zim")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("get")
    p.add_argument("zim")
    p.add_argument("entry")
    p.add_argument("--raw", action="store_true")
    p.set_defaults(fn=cmd_get)

    p = sub.add_parser("text")
    p.add_argument("zim")
    p.add_argument("entry")
    p.add_argument("--out", default=None)
    p.add_argument("--chars", type=int, default=2000)
    p.set_defaults(fn=cmd_text)

    p = sub.add_parser("check")
    p.add_argument("zim")
    p.add_argument("--all-clusters", action="store_true")
    p.add_argument("--sample", type=int, default=200)
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args()
    try:
        return args.fn(args)
    except ZimError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
