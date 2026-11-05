"""Extracción paralela en streaming del dump XML de Wikipedia.

Divide el archivo en rangos de bytes; cada proceso parsea las páginas que
*comienzan* en su rango y escribe tres tablas Parquet:
  pages_k.parquet : page_id, title, redirect   (solo ns=0)
  links_k.parquet : src_id, dst                (títulos destino normalizados)
  cats_k.parquet  : page_id, category

Uso: python src/extract_links.py XML OUT_DIR [--workers 12] [--limit-bytes N]
"""
import argparse
import html
import os
import re
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

LINK_RE = re.compile(rb"\[\[([^\[\]|#\n]{1,256})(?:[#|][^\[\]\n]*)?\]\]")
TITLE_RE = re.compile(rb"<title>(.*?)</title>")
NS_RE = re.compile(rb"<ns>(\d+)</ns>")
ID_RE = re.compile(rb"<id>(\d+)</id>")
REDIR_RE = re.compile(rb'<redirect title="(.*?)"')
FLUSH_EVERY = 2_000_000


def norm(t: str) -> str:
    """Normaliza un título como MediaWiki: espacios, primera letra mayúscula."""
    t = html.unescape(t).replace("_", " ").strip()
    t = " ".join(t.split())
    if t.startswith(":"):
        t = t[1:].lstrip()
    return t[:1].upper() + t[1:] if t else t


class Writer:
    def __init__(self, path, schema):
        self.w = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema
        self.cols = {n: [] for n in schema.names}

    def add(self, *vals):
        for n, v in zip(self.schema.names, vals):
            self.cols[n].append(v)
        if len(self.cols[self.schema.names[0]]) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if self.cols[self.schema.names[0]]:
            self.w.write_table(pa.table(self.cols, schema=self.schema))
            self.cols = {n: [] for n in self.schema.names}

    def close(self):
        self.flush()
        self.w.close()


def process_range(args):
    path, out_dir, k, start, end = args
    pages = Writer(f"{out_dir}/pages_{k:03d}.parquet", pa.schema(
        [("page_id", pa.int64()), ("title", pa.string()), ("redirect", pa.string())]))
    links = Writer(f"{out_dir}/links_{k:03d}.parquet", pa.schema(
        [("src_id", pa.int64()), ("dst", pa.string())]))
    cats = Writer(f"{out_dir}/cats_{k:03d}.parquet", pa.schema(
        [("page_id", pa.int64()), ("category", pa.string())]))
    n_pages = 0
    with open(path, "rb", buffering=16 << 20) as f:
        f.seek(start)
        if start:
            f.readline()  # descarta línea parcial
        pos = f.tell()
        in_page = False
        buf = []
        for line in f:
            if not in_page:
                if pos >= end:
                    break
                if line.lstrip().startswith(b"<page>"):
                    in_page, buf = True, []
                pos += len(line)
                continue
            pos += len(line)
            if line.lstrip().startswith(b"</page>"):
                in_page = False
                n_pages += _handle_page(b"".join(buf), pages, links, cats)
            else:
                buf.append(line)
    for w in (pages, links, cats):
        w.close()
    return k, n_pages


def _handle_page(raw, pages, links, cats):
    ns = NS_RE.search(raw)
    if ns is None or ns.group(1) != b"0":
        return 0
    title = norm(TITLE_RE.search(raw).group(1).decode("utf-8", "replace"))
    pid = int(ID_RE.search(raw).group(1))  # primer <id> = id de página
    r = REDIR_RE.search(raw)
    if r:
        pages.add(pid, title, norm(r.group(1).decode("utf-8", "replace")))
        return 1
    pages.add(pid, title, None)
    i = raw.find(b"<text")
    seen = set()
    for m in LINK_RE.finditer(raw, i):
        t = norm(m.group(1).decode("utf-8", "replace"))
        if not t or t in seen:
            continue
        seen.add(t)
        low = t.lower()
        if low.startswith("category:"):
            cats.add(pid, norm(t[9:]))
        elif ":" in t and low.split(":", 1)[0] in SKIP_PREFIXES:
            continue
        else:
            links.add(pid, t)
    return 1


SKIP_PREFIXES = {"file", "image", "wikipedia", "wp", "template", "help", "portal",
                 "user", "talk", "draft", "module", "special", "media", "wikt",
                 "wiktionary", "s", "commons", "d", "w", "mediawiki", "timedtext",
                 "user talk", "template talk", "category talk"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--chunks", type=int, default=96)
    ap.add_argument("--limit-bytes", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    size = a.limit_bytes or os.path.getsize(a.xml)
    step = -(-size // a.chunks)
    jobs = [(a.xml, a.out_dir, k, k * step, min(size, (k + 1) * step))
            for k in range(a.chunks)]
    t0, total = time.time(), 0
    with Pool(a.workers) as pool:
        for done, (k, n) in enumerate(pool.imap_unordered(process_range, jobs), 1):
            total += n
            print(f"[{time.time()-t0:7.0f}s] chunk {k:3d} listo ({n:,} artículos) "
                  f"{done}/{len(jobs)}  total={total:,}", flush=True)
    print("FIN", total, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
