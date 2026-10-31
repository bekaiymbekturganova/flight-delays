# Will this flight leave late?

Day-ahead prediction of US departure delays (15+ minutes) from public
[Bureau of Transportation Statistics](https://www.transtats.bts.gov/) on-time data.

The project is a small end-to-end system: download, clean, build features,
train, evaluate, publish a results page, and serve predictions over HTTP.

## The question

An operations planner looking at tomorrow's schedule wants to know which
departures are most likely to run late, early enough to move staff or gates.
So the model may only use what is known **the day before**: the published
schedule, the aircraft's planned rotation, the weather forecast, and delay
history up to yesterday. No same-day outcomes and no actual departure times.

## How it works

| Step | Module | What it does |
|---|---|---|
| Download | `download.py` | Fetches monthly files from BTS |
| Clean | `clean.py` | Keeps flights that departed, labels 15+ minute delays (DuckDB) |
| Weather | `weather.py` | Daily forecast weather per airport from Open-Meteo's forecast archive |
| Features | `features.py` | Schedule, 28-day trailing delay rates, the aircraft's scheduled rotation, weather at origin and destination |
| Train | `model.py` | Two baselines and LightGBM, split by time |
| Report | `report.py`, `dashboard.html` | Interactive results site: filter test-period predictions by day, airport, airline and time of day |
| Serve | `api.py` | FastAPI endpoint returning a delay probability |

## Decisions

- **Split by date, not at random.** Flights from the same afternoon share
  causes. A random split leaks that and inflates every metric. The model trains
  on the past and is tested on later weeks.
- **History windows end the day before the flight.** A test flips every
  outcome on one day and checks that the features for that day do not change
  (`tests/test_pipeline.py::test_no_leakage_from_the_flight_day`).
- **Baselines first.** The route's own recent delay rate is a strong, free
  predictor. The model has to beat it to be worth running.
- **PR-AUC and calibration, not accuracy.** Most flights leave on time, so
  "never delayed" scores high accuracy and helps nobody. The report shows
  whether a predicted 30% means roughly 30%.
- **Rotation from the schedule, not from what happened.** How late the
  aircraft's previous leg ran is the best single predictor of a delay, but it
  is not known the day before. The model uses what the schedule shows instead:
  which leg of the day this is for the aircraft and how long the planned
  turnaround is.
- **Forecast weather, not observed weather.** Weather comes from an archive of
  past model forecasts. Observed weather would be leakage.
- **Last week's trend, for seasonal swings.** The model trained on autumn and
  winter under-predicted summer delays. Seven-day delay rates (airport, airline
  and the whole US network) let it notice quickly when the system starts running
  late. Like every history feature, the window ends the day before the flight.
- **Re-centre the forecasts every day.** Even with the trend features the
  model's probabilities ran low in summer. Each evening the forecasts for
  tomorrow are shifted on the log-odds scale by the gap between the last 14
  days' forecasts and outcomes. Only finished days are used
  (`tests/test_pipeline.py::test_recalibration_only_uses_earlier_days`). This
  changes the level of the probabilities, not which flights look riskiest.
- **Every addition is measured.** The report keeps the schedule-and-history
  model next to the full one, so the gain from rotation, weather and the last-week trend is visible.
- **Cancelled flights stay in the schedule, but not in the labels.** The
  schedule features (traffic at the airport, the aircraft's legs and
  turnarounds) are computed on every scheduled flight, because cancellations
  are decided on the day, often because of storms. Counting only flights that
  operated would quietly tell the model "today was a bad day". Cancelled and
  diverted flights are left out of training and scoring: they have no departure
  delay, and cancellations have different causes.
  (`tests/test_pipeline.py::test_cancellations_on_the_flight_day_do_not_leak`)

## Ask the data in plain English

`flightdelays ask` turns a question into one DuckDB query with an LLM (Claude),
runs it, and prints the answer together with the SQL, so you can see exactly
what was counted.

```bash
pip install -e ".[ask]"
export ANTHROPIC_API_KEY=...        # keep it out of git, e.g. in .env
flightdelays ask "Which airline was late most often out of BOS in December?"
flightdelays ask-eval               # score it on the hand-checked question set
```

It is also on the API: `POST /ask {"question": "..."}`.

- **The checks matter more than the prompt.** Before anything runs, the SQL is
  parsed (sqlglot) and refused unless it is a single SELECT that reads only the
  `flights` table and only real columns. No writes, no other files, no
  `read_csv`, no environment variables.
- **DuckDB is locked down as well.** The connection may read only the one
  parquet file and its settings are locked, so even a query that got past the
  parser cannot open anything else
  (`tests/test_ask.py::test_connection_cannot_read_other_files`).
- **One retry with the error.** If a query is refused or fails, the error goes
  back to the model once. That fixes most small mistakes, like a wrong column name.
- **Says "can't answer" instead of guessing.** Questions the table cannot answer
  (weather, pilots) come back as a refusal with a reason, not invented SQL.
- **Measured, not eyeballed.** `ask_eval.json` holds 18 questions with
  hand-written SQL, including three that must be refused. An answer counts as
  correct when its rows match the hand-written query's rows (column names, row
  order and extra columns such as a flight count are ignored). The scoring
  itself is tested offline with a fake model, so CI needs no API key.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                   # runs on synthetic sample data
flightdelays all --start 2024-07 --end 2025-06
uvicorn flightdelays.api:app
```

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"carrier":"AA","origin":"LAX","dest":"BOS","flight_date":"2025-07-18","dep_hour":17}'
```

Or without installing anything: open the **Actions** tab, choose
**run pipeline**, and start it. It downloads the data, trains the model and
publishes the results page with GitHub Pages.

With Docker, after training:

```bash
docker build -t flightdelays .
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" flightdelays
```

## Results

Results are produced by the pipeline run and published on the results page.
They are not copied here by hand, so the numbers always match the code.

## Limits

- The forecast archive holds short-lead model runs, a slightly optimistic
  stand-in for a forecast issued exactly one day ahead.
- Aircraft assignments can change on the day. BTS records the tail number that
  actually flew, so after a same-day aircraft swap the rotation features
  describe the swapped aircraft. That is a small leak that this data cannot
  remove; the rotation gain should be read as an upper bound.
- Weather is daily and covers the 60 busiest airports, to stay inside the free
  weather API's limits. Other airports have no weather features.
- The API does not take rotation or weather inputs yet and treats them as missing.
- The serving snapshot uses the last four weeks of the training data. A real
  deployment would refresh it daily.
- Tests run on synthetic flights in the BTS layout. They check the logic, not
  model quality.
