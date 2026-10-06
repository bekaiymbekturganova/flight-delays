import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from flightdelays import model
from flightdelays.config import FEATURES, HISTORY_DAYS
from flightdelays.download import month_range
from flightdelays.report import write_report


def test_month_range_crosses_year():
    assert month_range("2024-11", "2025-02") == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    with pytest.raises(ValueError):
        month_range("2025-03", "2025-01")


def test_clean_drops_cancelled_and_labels_delay(workdir):
    raw = workdir["raw"]
    assert workdir["kept"] == int((raw.Cancelled == 0).sum())
    flights = pd.read_parquet(workdir["root"] / "flights.parquet")
    assert set(flights.delayed.unique()) <= {0, 1}
    assert (flights.delayed == (flights.dep_delay >= 15)).all()
    assert flights.dep_hour.between(0, 23).all()
    assert "ArrDelay" not in flights.columns


def test_features_keep_one_row_per_flight(workdir):
    flights = pd.read_parquet(workdir["root"] / "flights.parquet")
    feats = pd.read_parquet(workdir["root"] / "features.parquet")
    assert len(feats) == len(flights)
    assert set(FEATURES) <= set(feats.columns)


def test_history_matches_a_manual_calculation(workdir):
    """origin_rate must equal the delay rate over the 28 days before the flight."""
    flights = pd.read_parquet(workdir["root"] / "flights.parquet")
    feats = pd.read_parquet(workdir["root"] / "features.parquet")
    flights["flight_date"] = pd.to_datetime(flights.flight_date)
    feats["flight_date"] = pd.to_datetime(feats.flight_date)
    day = flights.flight_date.min() + pd.Timedelta(days=60)
    window = flights[
        (flights.origin == "ORD")
        & (flights.flight_date >= day - pd.Timedelta(days=HISTORY_DAYS))
        & (flights.flight_date < day)
    ]
    got = feats[(feats.origin == "ORD") & (feats.flight_date == day)]
    assert len(got) > 0
    assert np.allclose(got.origin_rate, window.delayed.mean())
    assert (got.origin_n == len(window)).all()


def test_no_leakage_from_the_flight_day(workdir, tmp_path):
    """Flipping every outcome on one day must not change that day's features."""
    src = workdir["root"] / "flights.parquet"
    flights = pd.read_parquet(src)
    day = sorted(flights.flight_date.unique())[70]
    tampered = flights.copy()
    mask = tampered.flight_date == day
    tampered.loc[mask, "delayed"] = 1 - tampered.loc[mask, "delayed"]
    tampered.to_parquet(tmp_path / "t.parquet")

    from flightdelays.features import build_features

    build_features(tmp_path / "t.parquet", tmp_path / "f.parquet")
    order = "carrier, origin, dest, dep_hour, distance, crs_elapsed"
    cols = ", ".join(FEATURES)
    con = duckdb.connect()
    a = con.execute(f"SELECT {cols} FROM '{workdir['root'] / 'features.parquet'}' WHERE flight_date = ? ORDER BY {order}", [day]).df()
    b = con.execute(f"SELECT {cols} FROM '{tmp_path / 'f.parquet'}' WHERE flight_date = ? ORDER BY {order}", [day]).df()
    pd.testing.assert_frame_equal(a, b)


def test_split_is_ordered_in_time(workdir):
    df = model.load(workdir["root"] / "features.parquet")
    train, valid, test = model.time_split(df)
    assert train.flight_date.max() < valid.flight_date.min()
    assert valid.flight_date.max() < test.flight_date.min()
    assert train.flight_date.min() >= df.flight_date.min() + pd.Timedelta(days=HISTORY_DAYS)


def test_train_beats_guessing_and_writes_artifacts(workdir):
    out = workdir["root"] / "artifacts"
    metrics = model.train(workdir["root"] / "features.parquet", out)
    gb = metrics["test_metrics"]["gradient_boosting"]
    assert gb["pr_auc"] > metrics["data"]["test_delay_rate"]
    assert gb["roc_auc"] > 0.55
    for name in ("model.txt", "categories.json", "metrics.json", "snapshot.parquet", "cube.json"):
        assert (out / name).exists()


def test_site_data_adds_up_to_the_test_period(workdir):
    out = workdir["root"] / "artifacts"
    if not (out / "cube.json").exists():
        model.train(workdir["root"] / "features.parquet", out)
    site = workdir["root"] / "site"
    assert write_report(out, site).exists()
    data = json.loads((site / "data.json").read_text())
    cube, test = data["cube"], data["metrics"]["data"]["test"]
    assert sum(r[4] for r in cube["rows"]) == test["flights"]
    late = sum(r[5] for r in cube["rows"]) / test["flights"]
    assert abs(late - data["metrics"]["data"]["test_delay_rate"]) < 1e-3
    assert cube["dates"][0] == test["from"] and cube["dates"][-1] == test["to"]
    assert all(0 <= r[6] <= r[4] for r in cube["rows"])


def test_api_predicts_a_probability(workdir, monkeypatch):
    from fastapi.testclient import TestClient

    from flightdelays import api

    out = workdir["root"] / "artifacts"
    if not (out / "model.txt").exists():
        model.train(workdir["root"] / "features.parquet", out)
    monkeypatch.setattr(api, "ARTIFACTS_DIR", out)
    api._load.cache_clear()
    client = TestClient(api.app)
    snap = pd.read_parquet(out / "snapshot.parquet").iloc[0]
    body = {"carrier": snap.carrier, "origin": snap.origin, "dest": snap.dest, "flight_date": "2025-06-02", "dep_hour": 18}
    res = client.post("/predict", json=body)
    assert res.status_code == 200
    assert 0 <= res.json()["delay_probability"] <= 1
    assert client.post("/predict", json={**body, "origin": "ZZZ"}).status_code == 404
    assert client.post("/predict", json={**body, "dep_hour": 30}).status_code == 422
    api._load.cache_clear()
