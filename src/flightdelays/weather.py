"""Daily forecast weather for every airport in the flight data.

Source: Open-Meteo's archive of past weather-model forecasts. This is what the
models said at short lead time, not what a forecaster published exactly one day
ahead, so it is a slightly optimistic stand-in for a real day-before forecast.
It is still much closer to one than observed weather would be.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

from .config import AIRPORTS_URL, CLEAN_PATH, WEATHER_PATH, WEATHER_URL, WEATHER_VARS

# Open-Meteo's free tier weighs a request by locations x days, and allows
# roughly 600 units a minute. A year of data for 10 airports is about 260 units,
# so requests are small and spaced out. Weather is fetched for the busiest
# airports only; the rest get missing values, which the model handles.
# The service is also slow at times, so a batch that keeps timing out is skipped
# rather than failing the whole pipeline: those airports just get missing values.
BATCH = 5
PAUSE_SECONDS = 15
TOP_AIRPORTS = 60
RETRIES = 4


def airport_coordinates(codes: list[str]) -> pd.DataFrame:
    """Latitude and longitude for IATA codes, from the OurAirports table."""
    response = requests.get(AIRPORTS_URL, timeout=120)
    response.raise_for_status()
    table = pd.read_csv(io.StringIO(response.text), usecols=["iata_code", "latitude_deg", "longitude_deg", "type"])
    table = table[table["iata_code"].isin(codes) & (table["type"] != "closed")]
    table = table.drop_duplicates("iata_code").rename(
        columns={"iata_code": "airport", "latitude_deg": "lat", "longitude_deg": "lon"}
    )
    return table[["airport", "lat", "lon"]].reset_index(drop=True)


def parse_response(payload, airports: list[str]) -> pd.DataFrame:
    """Turn an Open-Meteo response (one object per location) into long rows."""
    if isinstance(payload, dict):
        payload = [payload]
    if len(payload) != len(airports):
        raise ValueError(f"asked for {len(airports)} airports, got {len(payload)} back")
    frames = []
    for airport, item in zip(airports, payload):
        daily = item["daily"]
        frame = pd.DataFrame({"airport": airport, "date": pd.to_datetime(daily["time"]).date})
        for api_name, column in WEATHER_VARS.items():
            frame[column] = pd.to_numeric(pd.Series(daily[api_name]), errors="coerce")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fetch_batch(batch: pd.DataFrame, start: str, end: str, retries: int = RETRIES) -> pd.DataFrame | None:
    """One request for a few airports. Returns None if it keeps failing."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in batch["lat"]),
        "longitude": ",".join(f"{v:.4f}" for v in batch["lon"]),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(WEATHER_VARS),
        "timezone": "auto",  # days are local to each airport, like the flight dates
    }
    for attempt in range(retries):
        wait = 65 * (attempt + 1)
        try:
            response = requests.get(WEATHER_URL, params=params, timeout=180)
        except (requests.Timeout, requests.ConnectionError) as error:
            print(f"weather: {type(error).__name__}, retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            print(f"weather: HTTP {response.status_code}, retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        response.raise_for_status()  # a 4xx other than 429 is a bug in the request, so fail loudly
        return parse_response(response.json(), batch["airport"].tolist())
    print(f"weather: giving up on {', '.join(batch['airport'])}", flush=True)
    return None


def load_cache(path: Path, start, end) -> pd.DataFrame | None:
    """A weather file from an earlier run, if it covers the same dates.

    The pipeline keeps this file between runs, so each airport is fetched once
    and a run that skipped some airports fills them in next time.
    """
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if cached.empty or cached["date"].min() > start or cached["date"].max() < end:
        return None
    return cached[(cached["date"] >= start) & (cached["date"] <= end)]


def download_weather(
    flights: Path = CLEAN_PATH, out_path: Path = WEATHER_PATH, top: int = TOP_AIRPORTS
) -> int:
    """Fetch weather for the busiest airports over the dates in the flights table."""
    con = duckdb.connect()
    start, end = con.execute(f"SELECT min(flight_date), max(flight_date) FROM '{flights}'").fetchone()
    codes = [
        r[0]
        for r in con.execute(
            f"SELECT origin FROM '{flights}' GROUP BY origin ORDER BY count(*) DESC, origin LIMIT {int(top)}"
        ).fetchall()
    ]
    coords = airport_coordinates(codes)
    missing = sorted(set(codes) - set(coords["airport"]))
    if missing:
        print(f"weather: no coordinates for {len(missing)} airports: {', '.join(missing[:10])}", flush=True)
    frames = []
    cached = load_cache(out_path, start, end)
    if cached is not None:
        frames.append(cached)
        coords = coords[~coords["airport"].isin(cached["airport"])].reset_index(drop=True)
        print(f"weather: {cached['airport'].nunique()} airports from cache, {len(coords)} to fetch", flush=True)
    for i in range(0, len(coords), BATCH):
        frame = fetch_batch(coords.iloc[i : i + BATCH], str(start), str(end))
        if frame is not None:
            frames.append(frame)
        print(f"weather: {min(i + BATCH, len(coords))}/{len(coords)} airports done", flush=True)
        if i + BATCH < len(coords):
            time.sleep(PAUSE_SECONDS)
    got = sum(f["airport"].nunique() for f in frames)
    print(f"weather: have data for {got} airports", flush=True)
    if not frames:
        # No file means the feature step leaves the weather columns empty.
        out_path.unlink(missing_ok=True)
        return 0
    weather = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_path, index=False)
    return len(weather)
