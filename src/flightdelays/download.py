"""Download monthly on-time performance files from the BTS website."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

from .config import BTS_URL, RAW_DIR


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) between two 'YYYY-MM' strings."""
    y, m = (int(p) for p in start.split("-"))
    end_y, end_m = (int(p) for p in end.split("-"))
    if (y, m) > (end_y, end_m):
        raise ValueError(f"start {start} is after end {end}")
    months = []
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def download_month(year: int, month: int, out_dir: Path = RAW_DIR) -> Path:
    """Fetch one month and write its CSV to out_dir. Skips files already there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"ontime_{year}_{month:02d}.csv"
    if target.exists():
        return target
    url = BTS_URL.format(year=year, month=month)
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"expected one CSV in {url}, found {csv_names}")
        target.write_bytes(archive.read(csv_names[0]))
    return target


def download(start: str, end: str, out_dir: Path = RAW_DIR) -> list[Path]:
    paths = []
    for year, month in month_range(start, end):
        path = download_month(year, month, out_dir)
        print(f"{year}-{month:02d}: {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        paths.append(path)
    return paths
