"""Synthetic flights in the raw BTS layout, for tests only. Not real data."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

AIRPORTS = ["LAX", "BOS", "PHX", "MEM", "ORD", "DFW"]
CARRIERS = ["AA", "DL", "UA"]


def make_raw(days: int = 120, per_day: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * per_day
    day = np.repeat(np.arange(days), per_day)
    origin = rng.choice(AIRPORTS, n)
    dest = rng.choice(AIRPORTS, n)
    dest = np.where(dest == origin, np.roll(AIRPORTS, 1)[[AIRPORTS.index(o) for o in origin]], dest)
    hour = rng.integers(5, 24, n)
    carrier = rng.choice(CARRIERS, n)
    # Later departures and one congested airport run late more often.
    risk = 0.08 + 0.015 * (hour - 5) + 0.12 * (origin == "ORD") + 0.05 * (carrier == "UA")
    late = rng.random(n) < risk
    delay = np.where(late, rng.integers(15, 180, n), rng.integers(-10, 15, n)).astype(float)
    cancelled = (rng.random(n) < 0.02).astype(float)
    start = date(2025, 1, 1)
    return pd.DataFrame(
        {
            "FlightDate": [(start + timedelta(days=int(d))).isoformat() for d in day],
            "Reporting_Airline": carrier,
            "Origin": origin,
            "Dest": dest,
            "CRSDepTime": [f"{h:02d}{m:02d}" for h, m in zip(hour, rng.integers(0, 60, n))],
            "CRSElapsedTime": rng.integers(60, 360, n).astype(float),
            "Distance": rng.integers(200, 2600, n).astype(float),
            "DepDelay": np.where(cancelled == 1, np.nan, delay),
            "Cancelled": cancelled,
            "Diverted": 0.0,
            "ArrDelay": delay + rng.integers(-5, 5, n),
        }
    )


def write_raw(raw_dir: Path, **kwargs) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = make_raw(**kwargs)
    df.to_csv(raw_dir / "ontime_sample.csv", index=False)
    return df
