"""The question-to-SQL assistant. A fake model stands in for the API, so these
run offline and check the guard rails, the retry and the scoring, not the LLM."""
import json

import pandas as pd
import pytest

from flightdelays import ask
from flightdelays.ask import UnsafeQuery, check_sql


@pytest.fixture(scope="module")
def con(workdir):
    return ask.connect(workdir["root"] / "flights.parquet")


@pytest.mark.parametrize("sql", [
    "SELECT carrier, AVG(delayed) FROM flights GROUP BY carrier",
    "WITH b AS (SELECT * FROM flights WHERE origin = 'BOS') SELECT COUNT(*) AS n FROM b",
    "SELECT dayname(flight_date) AS d, COUNT(*) FROM flights GROUP BY d",
])
def test_read_only_queries_pass(sql):
    assert check_sql(sql) == sql


@pytest.mark.parametrize("sql", [
    "DELETE FROM flights",
    "DROP TABLE flights",
    "SELECT 1; DROP TABLE flights",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('data/features.parquet')",
    "SELECT * FROM other_table",
    "SELECT pilot_name FROM flights",
    "COPY flights TO 'out.csv'",
    "ATTACH 'x.db'",
    "SELECT getenv('ANTHROPIC_API_KEY')",
])
def test_unsafe_or_wrong_queries_are_refused(sql):
    with pytest.raises(UnsafeQuery):
        check_sql(sql)


def test_connection_cannot_read_other_files(con, tmp_path):
    (tmp_path / "secret.csv").write_text("a\n1\n")
    with pytest.raises(Exception):
        con.execute(f"SELECT * FROM read_csv('{tmp_path / 'secret.csv'}')").fetchall()


def test_answer_comes_with_rows_and_sql(con):
    sql = "SELECT carrier, COUNT(*) AS n FROM flights GROUP BY carrier ORDER BY carrier"
    a = ask.ask("flights per airline", lambda system, q: f"```sql\n{sql};\n```", con)
    assert a.refused is None and a.sql == sql
    assert list(a.table["carrier"]) == ["AA", "DL", "UA"]
    assert json.loads(json.dumps(a.as_dict()))["rows"]


def test_a_bad_query_is_retried_with_the_error(con):
    replies = iter(["SELECT pilot FROM flights", "SELECT COUNT(*) AS n FROM flights"])
    prompts = []

    def llm(system, prompt):
        prompts.append(prompt)
        return next(replies)

    a = ask.ask("how many flights?", llm, con)
    assert a.table is not None and "unknown column" in prompts[1]
    assert "corrected" in a.notes[0]


def test_unanswerable_questions_are_refused_not_guessed(con):
    a = ask.ask("weather tomorrow?", lambda s, q: "UNANSWERABLE The table has no weather.", con)
    assert a.table is None and "weather" in a.refused


def test_results_match_ignores_names_order_and_extra_columns():
    expected = pd.DataFrame({"carrier": ["UA", "AA"], "rate": [23.44, 18.0]})
    got = pd.DataFrame({"airline": ["AA", "UA"], "pct": [18.0, 23.4], "flights": [10, 20]})
    assert ask.results_match(got, expected)
    assert not ask.results_match(got.head(1), expected)


def test_eval_harness_scores_a_perfect_model_at_100(con):
    cases = json.loads(ask.EVAL_PATH.read_text())
    gold = {c["question"]: c.get("sql", "UNANSWERABLE not in the table") for c in cases}
    result = ask.evaluate(lambda system, q: gold[q], con)
    assert result["accuracy"] == 1.0, [c for c in result["cases"] if not c["ok"]]


def test_eval_harness_catches_wrong_answers(con):
    result = ask.evaluate(lambda system, q: "SELECT 1 AS x", con)
    assert result["accuracy"] < 0.2
