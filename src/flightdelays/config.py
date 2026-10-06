"""Paths, column names and constants shared by every step."""
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_PATH = DATA_DIR / "flights.parquet"
WEATHER_PATH = DATA_DIR / "weather.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.txt"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CATEGORIES_PATH = ARTIFACTS_DIR / "categories.json"

# Bureau of Transportation Statistics, "Reporting Carrier On-Time Performance".
BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Columns read from the raw BTS files. Anything recorded after departure
# (actual times, arrival delay, delay causes) is deliberately left out.
RAW_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Tail_Number",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "DepDelay",
    "Cancelled",
    "Diverted",
]

# A flight counts as delayed when it leaves the gate 15+ minutes late,
# the same threshold the US Department of Transportation reports.
DELAY_THRESHOLD_MIN = 15

# Trailing window for historical delay rates, in days.
HISTORY_DAYS = 28

# Public airport coordinates (OurAirports, public domain).
AIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
# Open-Meteo archive of past weather-model forecasts. Free, no key.
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_VARS = {
    "precipitation_sum": "precip_mm",
    "snowfall_sum": "snow_cm",
    "wind_gusts_10m_max": "gust_kmh",
    "temperature_2m_min": "temp_min_c",
}

TARGET = "delayed"
CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "dep_hour",
    "day_of_week",
    "month",
    "distance",
    "crs_elapsed",
    "origin_day_flights",
    "origin_hour_flights",
    "origin_rate",
    "origin_n",
    "carrier_rate",
    "carrier_n",
    "route_rate",
    "route_n",
    "origin_hour_rate",
    "origin_hour_n",
    # Aircraft rotation, from the schedule
    "leg_number",
    "legs_that_day",
    "turnaround_min",
    # Forecast weather at both ends of the flight
    "origin_precip_mm",
    "origin_snow_cm",
    "origin_gust_kmh",
    "origin_temp_min_c",
    "dest_precip_mm",
    "dest_snow_cm",
    "dest_gust_kmh",
    "dest_temp_min_c",
]
FEATURES = CATEGORICAL + NUMERIC

# The first version of the model used only schedule and history. It is kept as
# a comparison so the report shows what rotation and weather add.
ADDED = [f for f in NUMERIC if f in ("leg_number", "legs_that_day", "turnaround_min") or f.startswith(("origin_", "dest_")) and f.endswith(("_mm", "_cm", "_kmh", "_c"))]
BASE_FEATURES = [f for f in FEATURES if f not in ADDED]
