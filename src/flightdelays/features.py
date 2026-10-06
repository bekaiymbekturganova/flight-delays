"""Build model features. Every feature must be known the day before departure.

The risk in this kind of problem is leakage: using something about the day of
the flight (how late other flights ran, the weather that actually happened) to
"predict" it. Here the only inputs are the published schedule and delay history
that ends the day before the flight. tests/test_features.py checks that.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from .config import CLEAN_PATH, FEATURES_PATH, HISTORY_DAYS


def _history_cte(name: str, keys: list[str]) -> str:
    """Trailing delay rate per key over the HISTORY_DAYS before each date.

    The window ends one day before the current row, so a flight never sees
    outcomes from its own day.
    """
    cols = ", ".join(keys)
    return f"""
    {name}_daily AS (
        SELECT {cols}, flight_date, count(*) AS n, sum(delayed) AS d
        FROM flights GROUP BY {cols}, flight_date
    ),
    {name} AS (
        SELECT {cols}, flight_date,
            coalesce(sum(n) OVER w, 0) AS {name}_n,
            sum(d) OVER w / nullif(sum(n) OVER w, 0) AS {name}_rate
        FROM {name}_daily
        WINDOW w AS (
            PARTITION BY {cols} ORDER BY flight_date
            RANGE BETWEEN INTERVAL {HISTORY_DAYS} DAY PRECEDING
                      AND INTERVAL 1 DAY PRECEDING
        )
    )"""


def _join(name: str, keys: list[str]) -> str:
    on = " AND ".join(f"f.{k} = {name}.{k}" for k in keys)
    return f"LEFT JOIN {name} ON {on} AND f.flight_date = {name}.flight_date"


HISTORIES = {
    "origin": ["origin"],
    "carrier": ["carrier"],
    "route": ["origin", "dest"],
    "origin_hour": ["origin", "dep_hour"],
}


def build_query(source: str) -> str:
    ctes = ",".join(_history_cte(n, k) for n, k in HISTORIES.items())
    joins = "\n".join(_join(n, k) for n, k in HISTORIES.items())
    history_cols = ",\n        ".join(
        f"{n}.{n}_rate, {n}.{n}_n" for n in HISTORIES
    )
    return f"""
    WITH flights AS (SELECT * FROM '{source}'),
    {ctes}
    SELECT
        f.flight_date, f.carrier, f.origin, f.dest, f.dep_hour,
        isodow(f.flight_date) AS day_of_week,
        month(f.flight_date) AS month,
        f.distance, f.crs_elapsed,
        -- Scheduled traffic is published in advance, so it is fair to use.
        count(*) OVER (PARTITION BY f.origin, f.flight_date) AS origin_day_flights,
        count(*) OVER (PARTITION BY f.origin, f.flight_date, f.dep_hour) AS origin_hour_flights,
        {history_cols},
        f.dep_delay, f.delayed
    FROM flights f
    {joins}
    """


def build_features(source: Path = CLEAN_PATH, out_path: Path = FEATURES_PATH) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY ({build_query(str(source))}) TO '{out_path}' (FORMAT PARQUET)")
    return con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
