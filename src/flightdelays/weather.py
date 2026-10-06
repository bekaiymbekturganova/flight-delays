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

BATCH = 40  # airports per request


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


def fetch_batch(batch: pd.DataFrame, start: str, end: str, retries: int = 4) -> pd.DataFrame:
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in batch["lat"]),
        "longitude": ",".join(f"{v:.4f}" for v in batch["lon"]),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(WEATHER_VARS),
        "timezone": "auto",  # days are local to each airport, like the flight dates
    }
    for attempt in range(retries):
        response = requests.get(WEATHER_URL, params=params, timeout=120)
        if response.status_code == 429 and attempt < retries - 1:
            time.sleep(20 * (attempt + 1))  # rate limited: wait and try again
            continue
        response.raise_for_status()
        return parse_response(response.json(), batch["airport"].tolist())
    raise RuntimeError("weather request kept failing")


def download_weather(flights: Path = CLEAN_PATH, out_path: Path = WEATHER_PATH) -> int:
    """Fetch weather for all airports and dates in the cleaned flights table."""
    con = duckdb.connect()
    start, end = con.execute(f"SELECT min(flight_date), max(flight_date) FROM '{flights}'").fetchone()
    codes = [
        r[0]
        for r in con.execute(
            f"SELECT origin FROM '{flights}' UNION SELECT dest FROM '{flights}' ORDER BY 1"
        ).fetchall()
    ]
    coords = airport_coordinates(codes)
    missing = sorted(set(codes) - set(coords["airport"]))
    if missing:
        print(f"weather: no coordinates for {len(missing)} airports: {', '.join(missing[:10])}", flush=True)
    frames = []
    for i in range(0, len(coords), BATCH):
        frames.append(fetch_batch(coords.iloc[i : i + BATCH], str(start), str(end)))
        print(f"weather: {min(i + BATCH, len(coords))}/{len(coords)} airports", flush=True)
        time.sleep(1)
    weather = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_path, index=False)
    return len(weather)
