#!/usr/bin/env python
"""Download raw Oracle's Elixir match CSVs into data/raw/.

Sources (see docs/data.md):
  gdrive  - canonical per-year OE files via the Google Drive manifest
            (data/raw/oracleselixir_gdrive_manifest.tsv). Drive enforces a
            download quota; a quota-exceeded response is an HTML page which
            this script detects, deletes and warns about.
  hf      - combined 2014-2023 file mirrored on Hugging Face.
  mirror  - 2025 season file mirrored on GitHub.
  auto    - hf + mirror + gdrive for years neither covers (2024, 2026).

Idempotent: existing valid files are never re-downloaded.

Usage:
  .venv/bin/python scripts/download_data.py --dest data/raw --years 2014-2026 --source auto
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests

GDRIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id={id}&export=download&confirm=t"
)
HF_URL = (
    "https://huggingface.co/datasets/eligrayy/OE-LoL-Esports-Dataset/"
    "resolve/main/OE-LoL-Esports-Data.csv"
)
MIRROR_2025_URL = (
    "https://raw.githubusercontent.com/arthurcvl/LeagueCompetitiveStats/"
    "master/services/data-producer/newData.csv"
)
HF_YEARS = set(range(2014, 2024))
MIRROR_YEARS = {2025}
CHUNK = 1 << 20  # 1 MiB
PROGRESS_EVERY = 50 * (1 << 20)  # ~50 MB


def parse_years(spec: str) -> list[int]:
    """Parse '2014-2026' or '2019,2021' or '2024' into a sorted year list."""
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            years.update(range(int(lo), int(hi) + 1))
        elif part:
            years.add(int(part))
    return sorted(years)


def looks_like_html(head: bytes) -> bool:
    """True when the payload is an HTML page (Drive quota-exceeded, 404, ...)."""
    low = head[:2048].lstrip().lower()
    return low.startswith(b"<!doctype") or b"<html" in low or b"<!doctype" in low


def first_line_ok(path: Path) -> bool:
    """A valid raw OE CSV's first line starts with 'gameid'."""
    try:
        with open(path, "rb") as f:
            return f.readline().lstrip(b"\xef\xbb\xbf").startswith(b"gameid")
    except OSError:
        return False


def download(url: str, dest: Path, label: str) -> bool:
    """Stream url -> dest with progress; validate content. Returns success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[{label}] downloading {url}\n         -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            done = 0
            next_mark = PROGRESS_EVERY
            head = b""
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    if len(head) < 2048:
                        head += chunk[: 2048 - len(head)]
                    f.write(chunk)
                    done += len(chunk)
                    if done >= next_mark:
                        print(f"[{label}]   {done / 1e6:,.0f} MB so far")
                        next_mark += PROGRESS_EVERY
    except requests.RequestException as e:
        print(f"[{label}] WARNING: download failed ({e}); skipping", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    if looks_like_html(head):
        tmp.unlink(missing_ok=True)
        print(
            f"[{label}] WARNING: got an HTML page instead of a CSV — Google "
            "Drive download quota exceeded (or link changed). Deleted the "
            "file; retry later.",
            file=sys.stderr,
        )
        return False
    if not first_line_ok(tmp):
        tmp.unlink(missing_ok=True)
        print(
            f"[{label}] WARNING: first line does not start with 'gameid'; "
            "not an OE CSV. Deleted; skipping.",
            file=sys.stderr,
        )
        return False
    tmp.replace(dest)
    print(f"[{label}] OK: {dest} ({dest.stat().st_size / 1e6:,.1f} MB)")
    return True


def existing_valid(dest: Path, min_bytes: int = 1) -> bool:
    return (
        dest.is_file()
        and dest.stat().st_size >= min_bytes
        and first_line_ok(dest)
    )


def read_manifest(path: Path) -> dict[int, dict]:
    """Read the gdrive manifest TSV: year, file_id, bytes, filename."""
    if not path.is_file():
        print(f"WARNING: gdrive manifest not found at {path}", file=sys.stderr)
        return {}
    out: dict[int, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[int(row["year"])] = row
    return out


def fetch_gdrive(years: list[int], dest_dir: Path, manifest_path: Path) -> None:
    manifest = read_manifest(manifest_path)
    for year in years:
        entry = manifest.get(year)
        if entry is None:
            print(f"[gdrive] no manifest entry for {year}; skipping")
            continue
        dest = dest_dir / f"oe_{year}.csv"
        if existing_valid(dest):
            print(f"[gdrive] {dest} already present; skipping")
            continue
        download(GDRIVE_URL.format(id=entry["file_id"]), dest, "gdrive")


def fetch_hf(dest_dir: Path) -> None:
    dest = dest_dir / "oe_2014_2023.csv"
    if dest.is_file() and dest.stat().st_size > 400 * 1024 * 1024:
        print(f"[hf] {dest} already present (>400MB); skipping")
        return
    download(HF_URL, dest, "hf")


def fetch_mirror(dest_dir: Path) -> None:
    dest = dest_dir / "oe_2025.csv"
    if existing_valid(dest):
        print(f"[mirror] {dest} already present; skipping")
        return
    download(MIRROR_2025_URL, dest, "mirror")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default="data/raw", type=Path,
                    help="destination directory (default: data/raw)")
    ap.add_argument("--years", default="2014-2026",
                    help="year range/list, e.g. 2014-2026 or 2024,2026")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "gdrive", "hf", "mirror"],
                    help="which source(s) to pull from (default: auto)")
    args = ap.parse_args(argv)

    years = parse_years(args.years)
    dest_dir: Path = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "oracleselixir_gdrive_manifest.tsv"

    if args.source == "gdrive":
        fetch_gdrive(years, dest_dir, manifest_path)
    elif args.source == "hf":
        fetch_hf(dest_dir)
    elif args.source == "mirror":
        fetch_mirror(dest_dir)
    else:  # auto
        if any(y in HF_YEARS for y in years):
            fetch_hf(dest_dir)
        if any(y in MIRROR_YEARS for y in years):
            fetch_mirror(dest_dir)
        leftover = [y for y in years if y not in HF_YEARS | MIRROR_YEARS]
        if leftover:
            fetch_gdrive(leftover, dest_dir, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
