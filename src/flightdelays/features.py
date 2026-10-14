"""Build model features. Every feature must be known the day before departure.

The risk in this kind of problem is leakage: using something about the day of
the flight (how late other flights ran, how late the aircraft's previous leg
was, which flights got cancelled) to "predict" it. Here the inputs are the
published schedule (cancelled flights included), the aircraft's scheduled
rotation, forecast weather, and delay history that ends the day before the
flight. tests/test_pipeline.py checks that.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from .config import CLEAN_PATH, FEATURES_PATH, HISTORY_DAYS, RECENT_DAYS, WEATHER_PATH, WEATHER_VARS


def _history_cte(name: str, keys: list[str], days: int = HISTORY_DAYS) -> str:
    """Trailing delay rate per key over the `days` before each date.

    The window ends one day before the current row, so a flight never sees
    outcomes from its own day.
    """
    cols = ", ".join(keys)
    return f"""
    {name}_daily AS (
        SELECT {cols}, flight_date, count(delayed) AS n, sum(delayed) AS d
        FROM flights GROUP BY {cols}, flight_date
    ),
    {name} AS (
        SELECT {cols}, flight_date,
            coalesce(sum(n) OVER w, 0) AS {name}_n,
            sum(d) OVER w / nullif(sum(n) OVER w, 0) AS {name}_rate
        FROM {name}_daily
        WINDOW w AS (
            PARTITION BY {cols} ORDER BY flight_date
            RANGE BETWEEN INTERVAL {days} DAY PRECEDING
                      AND INTERVAL 1 DAY PRECEDING
        )
    )"""


def _join(name: str, keys: list[str]) -> str:
    on = " AND ".join(f"f.{k} = {name}.{k}" for k in keys)
    return f"LEFT JOIN {name} ON {on} AND f.flight_date = {name}.flight_date"


HISTORIES = {
    "origin": (["origin"], HISTORY_DAYS),
    "carrier": (["carrier"], HISTORY_DAYS),
    "route": (["origin", "dest"], HISTORY_DAYS),
    "origin_hour": (["origin", "dep_hour"], HISTORY_DAYS),
    # Last week only, so the model reacts quickly to seasonal swings.
    "origin_7d": (["origin"], RECENT_DAYS),
    "carrier_7d": (["carrier"], RECENT_DAYS),
    "nation_7d": (["nation"], RECENT_DAYS),
}


ROTATION_CTE = """
    rotation AS (
        -- Where each flight sits in its aircraft's scheduled day. A plane on its
        -- fifth leg with a 35-minute turnaround has little room to absorb a delay.
        -- Uses scheduled times only, so it is known the day before.
        SELECT
            flight_id,
            row_number() OVER tail_day AS leg_number,
            count(*) OVER (PARTITION BY tail, flight_date) AS legs_that_day,
            CASE
                WHEN lag(dest) OVER tail_day = origin
                 AND dep_min - lag(arr_min) OVER tail_day BETWEEN 0 AND 720
                THEN dep_min - lag(arr_min) OVER tail_day
            END AS turnaround_min
        FROM flights
        WHERE tail IS NOT NULL
        WINDOW tail_day AS (PARTITION BY tail, flight_date ORDER BY dep_min, flight_id)
    )"""


def _weather_sql(weather: str | None) -> tuple[str, str, str]:
    """CTE, joins and select list for weather at origin and destination."""
    names = list(WEATHER_VARS.values())
    if weather is None:
        cols = ", ".join(f"NULL::DOUBLE AS {end}_{n}" for end in ("origin", "dest") for n in names)
        return "", "", cols
    cte = f", weather AS (SELECT * FROM '{weather}')"
    joins = """
    LEFT JOIN weather wo ON wo.airport = f.origin AND wo.date = f.flight_date
    LEFT JOIN weather wd ON wd.airport = f.dest AND wd.date = f.flight_date"""
    cols = ", ".join(
        f"{alias}.{n} AS {end}_{n}" for end, alias in (("origin", "wo"), ("dest", "wd")) for n in names
    )
    return cte, joins, cols


def build_query(source: str, weather: str | None = None) -> str:
    ctes = ",".join(_history_cte(n, k, days) for n, (k, days) in HISTORIES.items())
    joins = "\n".join(_join(n, k) for n, (k, _) in HISTORIES.items())
    weather_cte, weather_joins, weather_cols = _weather_sql(weather)
    history_cols = ",\n        ".join(
        f"{n}.{n}_rate, {n}.{n}_n" for n in HISTORIES
    )
    return f"""
    WITH flights AS (
        -- Scheduled traffic is published in advance, so it is fair to use. It is
        -- counted here, before the WHERE at the bottom, because SQL applies WHERE
        -- before window functions: counting later would skip cancelled flights.
        SELECT *, 'US' AS nation,
            count(*) OVER (PARTITION BY origin, flight_date) AS origin_day_flights,
            count(*) OVER (PARTITION BY origin, flight_date, dep_hour) AS origin_hour_flights
        FROM '{source}'
    ),
    {ctes},
    {ROTATION_CTE}
    {weather_cte}
    SELECT
        f.flight_id, f.flight_date, f.carrier, f.origin, f.dest, f.dep_hour,
        isodow(f.flight_date) AS day_of_week,
        month(f.flight_date) AS month,
        f.distance, f.crs_elapsed,
        f.origin_day_flights, f.origin_hour_flights,
        {history_cols},
        rotation.leg_number, rotation.legs_that_day, rotation.turnaround_min,
        {weather_cols},
        f.dep_delay, f.delayed
    FROM flights f
    {joins}
    LEFT JOIN rotation ON rotation.flight_id = f.flight_id
    {weather_joins}
    -- Schedule features above are computed on every scheduled flight; only
    -- flights that actually operated have a label to learn from.
    WHERE f.operated
    """


def build_features(
    source: Path = CLEAN_PATH,
    out_path: Path = FEATURES_PATH,
    weather: Path | None = WEATHER_PATH,
) -> int:
    """Write the feature table. Weather columns are empty if no weather file exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather_arg = str(weather) if weather is not None and Path(weather).exists() else None
    con = duckdb.connect()
    con.execute(f"COPY ({build_query(str(source), weather_arg)}) TO '{out_path}' (FORMAT PARQUET)")
    return con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
