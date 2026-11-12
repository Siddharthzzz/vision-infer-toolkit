"""Parallel, resumable HTTP range downloader (stdlib only).

Both dataset hosts throttle each connection (measured 2026-11-11: COCO about 30-70 kB/s per
connection, about 355 kB/s total with 8 connections), so one file is split into byte ranges
fetched concurrently. Finished chunks are kept in <dest>.parts/, so re-running resumes.

usage: python3 -I scripts/fetch.py URL DEST [--workers 16] [--chunk-mb 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import shutil
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "vision-infer-toolkit-fetch/0.1"}


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            sys.exit(f"server does not advertise byte ranges: {url}")
        return int(r.headers["Content-Length"])


def fetch_chunk(url: str, part: Path, start: int, end: int, retries: int = 8) -> int:
    want = end - start + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return want
        if have > want:
            part.unlink()
            have = 0
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, part.open("ab") as f:
                if r.status != 206:
                    raise OSError(f"expected 206, got {r.status}")
                shutil.copyfileobj(r, f, 1 << 20)
        except OSError as e:
            print(f"  retry {attempt + 1} {part.name}: {e}", file=sys.stderr)
            time.sleep(min(60, 2**attempt))
    if part.stat().st_size != want:
        raise RuntimeError(f"chunk {part.name} incomplete after {retries} retries")
    return want


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=8)
    a = ap.parse_args()

    total = content_length(a.url)
    if a.dest.exists() and a.dest.stat().st_size == total:
        print(f"{a.dest} already complete ({total} bytes)")
        return
    parts = a.dest.with_name(a.dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    step = a.chunk_mb << 20
    ranges = [(i, s, min(s + step, total) - 1) for i, s in enumerate(range(0, total, step))]

    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch_chunk, a.url, parts / f"{i:05d}", s, e) for i, s, e in ranges]
        for k, fut in enumerate(cf.as_completed(futs), 1):
            done += fut.result()
            if k % 10 == 0 or k == len(futs):
                rate = done / max(time.time() - t0, 1e-9) / 1e3
                print(f"  {a.dest.name}: {k}/{len(futs)} chunks, {rate:.0f} kB/s", flush=True)

    tmp = a.dest.with_name(a.dest.name + ".tmp")
    with tmp.open("wb") as out:
        for i, _, _ in ranges:
            with (parts / f"{i:05d}").open("rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
    if tmp.stat().st_size != total:
        sys.exit(f"size mismatch: {tmp.stat().st_size} != {total}")
    tmp.rename(a.dest)
    shutil.rmtree(parts)
    print(f"{a.dest}: {total} bytes OK")


if __name__ == "__main__":
    main()
