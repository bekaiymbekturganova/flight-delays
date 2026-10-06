"""HTTP service: delay probability for a scheduled flight.

Run with:  uvicorn flightdelays.api:app
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ARTIFACTS_DIR, CATEGORIES_PATH, FEATURES, MODEL_PATH
from .model import SNAPSHOT_PATH, to_matrix

app = FastAPI(title="Flight delay risk", version="0.1.0")


class Flight(BaseModel):
    carrier: str = Field(examples=["AA"], description="IATA carrier code")
    origin: str = Field(examples=["LAX"], description="Origin airport code")
    dest: str = Field(examples=["BOS"], description="Destination airport code")
    flight_date: date
    dep_hour: int = Field(ge=0, le=23, description="Scheduled local departure hour")


class Prediction(BaseModel):
    delay_probability: float
    route_history_rate: float | None
    note: str


@lru_cache
def _load():
    if not (ARTIFACTS_DIR / MODEL_PATH.name).exists():
        raise FileNotFoundError("no trained model; run `flightdelays all` first")
    booster = lgb.Booster(model_file=str(ARTIFACTS_DIR / MODEL_PATH.name))
    categories = json.loads((ARTIFACTS_DIR / CATEGORIES_PATH.name).read_text())
    snapshot = pd.read_parquet(ARTIFACTS_DIR / SNAPSHOT_PATH.name)
    return booster, categories, snapshot


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(flight: Flight) -> Prediction:
    try:
        booster, categories, snapshot = _load()
    except FileNotFoundError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    match = snapshot[
        (snapshot.carrier == flight.carrier)
        & (snapshot.origin == flight.origin)
        & (snapshot.dest == flight.dest)
    ]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"{flight.carrier} did not fly {flight.origin}-{flight.dest} in the last four weeks of data",
        )
    # Use the closest scheduled hour this carrier flies on the route.
    row = match.iloc[(match.dep_hour - flight.dep_hour).abs().argsort().iloc[0]].to_dict()
    exact_hour = row["dep_hour"] == flight.dep_hour
    row.update(
        dep_hour=flight.dep_hour,
        day_of_week=flight.flight_date.isoweekday(),
        month=flight.flight_date.month,
    )
    # Aircraft rotation and weather are not known to this endpoint yet. LightGBM
    # treats them as missing, so the answer leans on schedule and history.
    for name in FEATURES:
        row.setdefault(name, None)
    X = to_matrix(pd.DataFrame([row]), categories)
    probability = float(booster.predict(X)[0])
    rate = row.get("route_rate")
    return Prediction(
        delay_probability=round(probability, 4),
        route_history_rate=None if pd.isna(rate) else round(float(rate), 4),
        note="exact scheduled hour" if exact_hour else "nearest scheduled hour on this route used for history",
    )
