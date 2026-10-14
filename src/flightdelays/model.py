"""Train and evaluate: two baselines and a gradient-boosted model, split by time."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ARTIFACTS_DIR,
    BASE_FEATURES,
    V2_FEATURES,
    CATEGORICAL,
    CATEGORIES_PATH,
    FEATURES,
    FEATURES_PATH,
    HISTORY_DAYS,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC,
    RECAL_DAYS,
    TARGET,
)

SNAPSHOT_PATH = ARTIFACTS_DIR / "snapshot.parquet"
CUBE_PATH = ARTIFACTS_DIR / "cube.json"

# Time-of-day buckets used on the results site, by scheduled local hour.
DAYPARTS = ["Morning (5-11)", "Midday (11-16)", "Evening (16-21)", "Night (21-5)"]
TOP_AIRPORTS = 30
OTHER = "Other airports"


def daypart(hour: pd.Series) -> pd.Series:
    return pd.cut(
        (hour - 5) % 24, bins=[-1, 5, 10, 15, 23], labels=range(4)
    ).astype(int)


def prediction_cube(test_df: pd.DataFrame, p_test: np.ndarray) -> dict:
    """Test-period predictions aggregated for the results site.

    One row per (day, airport, carrier, time of day) with the number of
    flights, how many were late, and the sum of predicted probabilities.
    Small airports are grouped so the file stays small.
    """
    frame = pd.DataFrame(
        {
            "date": test_df["flight_date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "origin": test_df["origin"].to_numpy(),
            "carrier": test_df["carrier"].to_numpy(),
            "part": daypart(test_df["dep_hour"]).to_numpy(),
            "late": test_df[TARGET].to_numpy(),
            "p": p_test,
        }
    )
    top = frame["origin"].value_counts().head(TOP_AIRPORTS).index
    frame["origin"] = frame["origin"].where(frame["origin"].isin(top), OTHER)
    dates = sorted(frame["date"].unique())
    origins = sorted(o for o in frame["origin"].unique() if o != OTHER)
    if (frame["origin"] == OTHER).any():
        origins.append(OTHER)
    carriers = sorted(frame["carrier"].unique())
    grouped = (
        frame.groupby(["date", "origin", "carrier", "part"], observed=True)
        .agg(n=("late", "size"), late=("late", "sum"), p=("p", "sum"))
        .reset_index()
    )
    d_ix = {v: i for i, v in enumerate(dates)}
    o_ix = {v: i for i, v in enumerate(origins)}
    c_ix = {v: i for i, v in enumerate(carriers)}
    rows = [
        [d_ix[r.date], o_ix[r.origin], c_ix[r.carrier], int(r.part), int(r.n), int(r.late), round(float(r.p), 2)]
        for r in grouped.itertuples()
    ]
    return {"dates": dates, "origins": origins, "carriers": carriers, "parts": DAYPARTS, "rows": rows}


def load(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = duckdb.connect().execute(f"SELECT * FROM '{path}'").df()
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


def time_split(df: pd.DataFrame, valid_frac: float = 0.1, test_frac: float = 0.2):
    """Split by calendar date: train on the past, validate and test on the future.

    A random split would put flights from the same stormy afternoon in both
    train and test and make the model look better than it is. The first
    HISTORY_DAYS days are dropped because their history features are incomplete.
    """
    start = df["flight_date"].min() + timedelta(days=HISTORY_DAYS)
    df = df[df["flight_date"] >= start]
    days = np.sort(df["flight_date"].unique())
    if len(days) < 10:
        raise ValueError("need at least 10 days of data after the history warm-up")
    test_start = days[int(len(days) * (1 - test_frac))]
    valid_start = days[int(len(days) * (1 - test_frac - valid_frac))]
    train = df[df["flight_date"] < valid_start]
    valid = df[(df["flight_date"] >= valid_start) & (df["flight_date"] < test_start)]
    test = df[df["flight_date"] >= test_start]
    return train, valid, test


def fit_categories(train: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL}


def to_matrix(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Model input. Airports or carriers not seen in training become missing."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrate(dates: np.ndarray, y: np.ndarray, p: np.ndarray, score_from, window: int = RECAL_DAYS) -> np.ndarray:
    """Shift each day's probabilities so recent forecasts match recent outcomes.

    The model learns from autumn and winter and is scored on summer, when more
    flights run late, so its raw probabilities come out too low. Every morning
    (in the evening before, really) we already know how the last `window` days
    turned out. For each day D from `score_from` on, the probabilities are moved
    on the log-odds scale by one number:

        shift = logit(actual late rate) - logit(average forecast)

    both measured over the days D-window .. D-1. Only days strictly before D are
    used, so no outcome from the flight's own day leaks in. The order of flights
    within a day does not change; only the level does.

    Returns the adjusted probabilities for the rows on or after `score_from`.
    """
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy()
    frame = pd.DataFrame({"d": dates, "y": y, "p": p})
    daily = frame.groupby("d").agg(late=("y", "sum"), p=("p", "sum"), n=("y", "size"))
    out_mask = dates >= np.datetime64(pd.Timestamp(score_from))
    shift = {}
    for day in np.unique(dates[out_mask]):
        past = daily.loc[(daily.index < day) & (daily.index >= day - np.timedelta64(window, "D"))]
        if past["n"].sum() == 0:
            shift[day] = 0.0
            continue
        obs, pred = past["late"].sum() / past["n"].sum(), past["p"].sum() / past["n"].sum()
        shift[day] = float(_logit(np.array([obs]))[0] - _logit(np.array([pred]))[0])
    s = np.array([shift[d] for d in dates[out_mask]])
    return 1 / (1 + np.exp(-(_logit(p[out_mask]) + s)))


def score(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Predicted vs. observed delay rate in equal-sized groups of flights."""
    frame = pd.DataFrame({"y": y_true, "p": p})
    frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    grouped = frame.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), flights=("y", "size"))
    return [
        {"predicted": round(r.predicted, 4), "observed": round(r.observed, 4), "flights": int(r.flights)}
        for r in grouped.itertuples()
    ]


def train(features_path: Path = FEATURES_PATH, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    df = load(features_path)
    train_df, valid_df, test_df = time_split(df)
    categories = fit_categories(train_df)
    X_train, X_valid, X_test = (to_matrix(d, categories) for d in (train_df, valid_df, test_df))
    y_train, y_valid, y_test = (d[TARGET].to_numpy() for d in (train_df, valid_df, test_df))
    base_rate = float(y_train.mean())

    results = {}

    # Baseline 1: the route's own delay rate over the last four weeks.
    route = test_df["route_rate"].fillna(test_df["origin_rate"]).fillna(base_rate).to_numpy()
    results["route_history"] = score(y_test, route)

    # Baseline 2: logistic regression on the numeric features.
    logit = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500)
    )
    logit.fit(X_train[NUMERIC], y_train)
    results["logistic_regression"] = score(y_test, logit.predict_proba(X_test[NUMERIC])[:, 1])

    # Gradient boosting, stopped early on the validation period.
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "cat_smooth": 20,
        "verbose": -1,
        "seed": 7,
    }

    def fit(columns: list[str]) -> lgb.Booster:
        return lgb.train(
            params,
            lgb.Dataset(X_train[columns], y_train),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_valid[columns], y_valid)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

    # Same model without rotation and weather, to measure what they add.
    base = fit(BASE_FEATURES)
    results["gradient_boosting_base"] = score(
        y_test, base.predict(X_test[BASE_FEATURES], num_iteration=base.best_iteration)
    )
    v2 = fit(V2_FEATURES)
    results["gradient_boosting_v2"] = score(
        y_test, v2.predict(X_test[V2_FEATURES], num_iteration=v2.best_iteration)
    )
    booster = fit(FEATURES)
    p_raw = booster.predict(X_test, num_iteration=booster.best_iteration)
    results["gradient_boosting"] = score(y_test, p_raw)

    # Re-centre each day's forecasts on the last two weeks of known outcomes.
    # The validation period supplies the history for the first test days.
    p_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)
    p_test = recalibrate(
        np.concatenate([valid_df["flight_date"].to_numpy(), test_df["flight_date"].to_numpy()]),
        np.concatenate([y_valid, y_test]),
        np.concatenate([p_valid, p_raw]),
        score_from=test_df["flight_date"].min(),
    )
    results["gradient_boosting_recalibrated"] = score(y_test, p_test)

    by_hour = (
        pd.DataFrame({"dep_hour": test_df["dep_hour"].to_numpy(), "y": y_test, "p": p_test})
        .groupby("dep_hour")
        .agg(observed=("y", "mean"), predicted=("p", "mean"), flights=("y", "size"))
        .round(4)
        .reset_index()
        .to_dict("records")
    )
    importance = sorted(
        zip(booster.feature_name(), booster.feature_importance("gain").tolist()),
        key=lambda kv: -kv[1],
    )
    total_gain = sum(v for _, v in importance) or 1.0

    metrics = {
        "data": {
            "first_day": str(df["flight_date"].min().date()),
            "last_day": str(df["flight_date"].max().date()),
            "train": {"flights": len(train_df), "from": str(train_df["flight_date"].min().date()), "to": str(train_df["flight_date"].max().date())},
            "valid": {"flights": len(valid_df), "from": str(valid_df["flight_date"].min().date()), "to": str(valid_df["flight_date"].max().date())},
            "test": {"flights": len(test_df), "from": str(test_df["flight_date"].min().date()), "to": str(test_df["flight_date"].max().date())},
            "train_delay_rate": round(base_rate, 4),
            "test_delay_rate": round(float(y_test.mean()), 4),
        },
        "test_metrics": results,
        "best_iteration": int(booster.best_iteration or 0),
        "calibration": calibration_table(y_test, p_test),
        "calibration_raw": calibration_table(y_test, p_raw),
        "recalibration": {
            "window_days": RECAL_DAYS,
            "mean_raw": round(float(p_raw.mean()), 4),
            "mean_recalibrated": round(float(p_test.mean()), 4),
            "observed": round(float(y_test.mean()), 4),
        },
        "by_hour": by_hour,
        "importance": [{"feature": f, "share": round(v / total_gain, 4)} for f, v in importance],
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(artifacts_dir / MODEL_PATH.name), num_iteration=booster.best_iteration)
    (artifacts_dir / CATEGORIES_PATH.name).write_text(json.dumps(categories))
    (artifacts_dir / METRICS_PATH.name).write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / CUBE_PATH.name).write_text(
        json.dumps(prediction_cube(test_df, p_test), separators=(",", ":"))
    )
    save_snapshot(df, artifacts_dir / SNAPSHOT_PATH.name)
    return metrics


def save_snapshot(df: pd.DataFrame, path: Path) -> None:
    """Latest history and typical schedule load per key, for serving predictions."""
    last_day = df["flight_date"].max()
    recent = df[df["flight_date"] > last_day - timedelta(days=HISTORY_DAYS)]
    keys = ["carrier", "origin", "dest", "dep_hour"]
    snap = (
        recent.sort_values("flight_date")
        .groupby(keys, as_index=False)
        .agg(
            distance=("distance", "median"),
            crs_elapsed=("crs_elapsed", "median"),
            origin_day_flights=("origin_day_flights", "median"),
            origin_hour_flights=("origin_hour_flights", "median"),
            **{c: (c, "last") for c in NUMERIC if c.endswith(("_rate", "_n"))},
        )
    )
    snap.to_parquet(path, index=False)
