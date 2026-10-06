"""Turn raw BTS CSV files into one tidy table of flights that departed."""
from __future__ import annotations

from pathlib import Path

import duckdb

from .config import CLEAN_PATH, DELAY_THRESHOLD_MIN, RAW_COLUMNS, RAW_DIR

CLEAN_SQL = """
COPY (
    SELECT
        CAST(FlightDate AS DATE)                       AS flight_date,
        Reporting_Airline                              AS carrier,
        Origin                                         AS origin,
        Dest                                           AS dest,
        nullif(trim(Tail_Number), '')                  AS tail,
        -- Scheduled local times as minutes after midnight.
        (TRY_CAST(CRSDepTime AS INTEGER) // 100) * 60 + TRY_CAST(CRSDepTime AS INTEGER) % 100 AS dep_min,
        (TRY_CAST(CRSArrTime AS INTEGER) // 100) * 60 + TRY_CAST(CRSArrTime AS INTEGER) % 100 AS arr_min,
        -- CRSDepTime is local scheduled time as an hhmm integer; 2400 means midnight.
        CAST(floor(TRY_CAST(CRSDepTime AS INTEGER) / 100) AS INTEGER) % 24 AS dep_hour,
        TRY_CAST(CRSElapsedTime AS DOUBLE)                 AS crs_elapsed,
        TRY_CAST(Distance AS DOUBLE)                       AS distance,
        TRY_CAST(DepDelay AS DOUBLE)                   AS dep_delay,
        CAST(TRY_CAST(DepDelay AS DOUBLE) >= {threshold} AS INTEGER) AS delayed
    FROM read_csv(
        '{glob}',
        header = true,
        union_by_name = true,
        all_varchar = true,
        ignore_errors = true
    )
    WHERE TRY_CAST(Cancelled AS DOUBLE) = 0
      AND TRY_CAST(Diverted AS DOUBLE) = 0
      AND TRY_CAST(DepDelay AS DOUBLE) IS NOT NULL
      AND TRY_CAST(CRSDepTime AS INTEGER) IS NOT NULL
      AND Origin IS NOT NULL
      AND Dest IS NOT NULL
) TO '{out}' (FORMAT PARQUET)
"""


def clean(raw_dir: Path = RAW_DIR, out_path: Path = CLEAN_PATH) -> int:
    """Write the cleaned parquet file and return the number of flights kept.

    Cancelled and diverted flights are dropped: they have no departure delay,
    and predicting cancellations is a different problem with different causes.
    """
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {raw_dir}; run the download step first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    header = con.execute(
        f"SELECT * FROM read_csv('{files[0]}', header=true, all_varchar=true) LIMIT 0"
    ).df().columns
    missing = [c for c in RAW_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{files[0]} is missing expected columns: {missing}")
    con.execute(
        CLEAN_SQL.format(
            threshold=DELAY_THRESHOLD_MIN,
            glob=str(raw_dir / "*.csv"),
            out=str(out_path),
        )
    )
    return con.execute(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
