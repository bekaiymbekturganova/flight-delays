"""Paths, column names and constants shared by every step."""
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_PATH = DATA_DIR / "flights.parquet"
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
    "CRSDepTime",
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
]
FEATURES = CATEGORICAL + NUMERIC
