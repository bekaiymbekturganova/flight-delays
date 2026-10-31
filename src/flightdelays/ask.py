"""Ask the flight data a question in plain English.

    flightdelays ask "Which airline was late most often out of BOS in December?"

A language model writes one DuckDB query, the query is checked before it runs,
and the answer comes back with the SQL so anyone can see what was counted.

The checks matter more than the prompt. The model may only read the `flights`
table, may only use columns that exist, and gets exactly one SELECT. Anything
else (writes, other files, several statements) is refused before DuckDB sees
it. Results are capped, so a careless query cannot dump the whole table.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from .config import CLEAN_PATH

TABLE = "flights"
MAX_ROWS = 200
DEFAULT_MODEL = os.environ.get("FLIGHTDELAYS_LLM_MODEL", "claude-sonnet-5-5")

# What the model is told about the table. Kept next to the checks so the two
# cannot drift apart: every column listed here is the full allowed set.
COLUMNS = {
    "flight_id": "row id",
    "flight_date": "DATE, scheduled day of the flight",
    "carrier": "airline IATA code, e.g. AA, DL, UA, WN, B6",
    "origin": "departure airport IATA code, e.g. BOS, LAX",
    "dest": "arrival airport IATA code",
    "tail": "aircraft tail number (may be NULL)",
    "dep_min": "scheduled departure, minutes after local midnight",
    "arr_min": "scheduled arrival, minutes after local midnight",
    "dep_hour": "scheduled departure hour, 0-23 local time",
    "crs_elapsed": "scheduled flight time in minutes",
    "distance": "distance in miles",
    "operated": "BOOLEAN, false if the flight was cancelled or diverted",
    "dep_delay": "departure delay in minutes (negative = early); NULL if not operated",
    "delayed": "1 if the flight left 15+ minutes late, 0 if not, NULL if not operated",
}

SYSTEM_PROMPT = f"""You turn questions about US flights into one DuckDB SQL query.

There is one table, `{TABLE}`, one row per scheduled flight
(Bureau of Transportation Statistics on-time data):

{chr(10).join(f"- {c}: {d}" for c, d in COLUMNS.items())}

Rules:
- Write exactly one SELECT statement for DuckDB. Read only from `{TABLE}`.
- "Delay rate" or "how often late" means AVG(delayed) over operated flights,
  reported as a percentage rounded to 1 decimal.
- Cancellation rate means AVG(CASE WHEN operated THEN 0 ELSE 1 END).
- Use only the columns above. Airport and airline codes are upper case.
- When ranking groups, also return the number of flights so small groups are visible.
- If the question cannot be answered from this table, return the word
  UNANSWERABLE and one short sentence saying why, instead of SQL.

Reply with the SQL only, no explanation, no markdown fences."""


class UnsafeQuery(ValueError):
    """The generated SQL was refused before running."""


@dataclass
class Answer:
    question: str
    sql: str | None
    table: pd.DataFrame | None = None
    refused: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "sql": self.sql,
            "refused": self.refused,
            "columns": [] if self.table is None else list(self.table.columns),
            "rows": [] if self.table is None else json.loads(self.table.to_json(orient="values", date_format="iso")),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Checking the query
# --------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Models sometimes wrap SQL in ```sql fences despite being asked not to."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def check_sql(sql: str) -> str:
    """Return the SQL if it is one read-only query on `flights`, else raise UnsafeQuery."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeQuery(f"could not parse the SQL: {e}") from None
    if len(statements) != 1:
        raise UnsafeQuery("expected exactly one statement")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeQuery(f"only SELECT is allowed, got {type(tree).__name__}")

    banned = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
              exp.Alter, exp.Command, exp.Copy, exp.Attach, exp.Pragma)
    for node in tree.walk():
        if isinstance(node, banned):
            raise UnsafeQuery(f"{type(node).__name__} is not allowed")
        # read_csv('...'), read_parquet(...) and friends would reach other files.
        if isinstance(node, (exp.Anonymous, exp.Func)) and re.match(
            r"(read_|glob|sniff_|parquet_|query_table|getenv)", node.sql(dialect="duckdb").lower()
        ):
            raise UnsafeQuery("functions that read files or the environment are not allowed")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name:  # a table function such as read_csv(...)
            raise UnsafeQuery("table functions are not allowed")
        if name != TABLE and name not in ctes:
            raise UnsafeQuery(f"unknown table: {table.name}")

    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)} | ctes
    known = set(COLUMNS)
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if name and name not in known and name not in aliases and name != "*":
            raise UnsafeQuery(f"unknown column: {col.name}")
    return sql


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------

def connect(parquet: Path = CLEAN_PATH) -> duckdb.DuckDBPyConnection:
    if not Path(parquet).exists():
        raise FileNotFoundError(f"{parquet} not found; run `flightdelays clean` first")
    path = Path(parquet).resolve().as_posix()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW {TABLE} AS SELECT * FROM read_parquet('{path}')")
    # From here on queries can read only this one file: no other files,
    # no extensions, and the settings themselves are locked.
    con.execute(f"SET allowed_paths = ['{path}']")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def run_sql(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, bool]:
    """Run a checked query; return the rows (capped) and whether they were cut off."""
    check_sql(sql)
    df = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {max_rows + 1}").df()
    return df.head(max_rows), len(df) > max_rows


# --------------------------------------------------------------------------
# The language model
# --------------------------------------------------------------------------

LLM = Callable[[str, str], str]  # (system prompt, question) -> reply text


def anthropic_llm(model: str = DEFAULT_MODEL) -> LLM:
    """Claude via the Anthropic API. Reads ANTHROPIC_API_KEY from the environment."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError('install the extra first:  pip install -e ".[ask]"') from None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("set ANTHROPIC_API_KEY (for example in a .env file you do not commit)")
    client = anthropic.Anthropic()

    def call(system: str, question: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=800, system=system,
            messages=[{"role": "user", "content": question}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    return call


def ask(question: str, llm: LLM, con: duckdb.DuckDBPyConnection, retries: int = 1) -> Answer:
    """Question -> checked SQL -> rows. On a refused or failing query, retry once
    with the error shown to the model, which fixes most small mistakes."""
    prompt = question
    sql = None
    for attempt in range(retries + 1):
        reply = strip_fences(llm(SYSTEM_PROMPT, prompt))
        if reply.upper().startswith("UNANSWERABLE"):
            return Answer(question, None, refused=reply.split(None, 1)[1] if " " in reply else "not answerable from this table")
        sql = reply
        try:
            table, cut = run_sql(con, sql)
        except (UnsafeQuery, duckdb.Error) as e:
            if attempt == retries:
                return Answer(question, sql, refused=f"query rejected: {e}")
            prompt = f"{question}\n\nYour previous SQL:\n{sql}\n\nfailed with: {e}\nWrite a corrected query."
            continue
        notes = [f"showing the first {MAX_ROWS} rows"] if cut else []
        if attempt:
            notes.append("the first query failed and was corrected")
        return Answer(question, sql, table, notes=notes)
    return Answer(question, sql, refused="no answer")  # pragma: no cover


# --------------------------------------------------------------------------
# Measuring it
# --------------------------------------------------------------------------

EVAL_PATH = Path(__file__).with_name("ask_eval.json")


def _normalise(df: pd.DataFrame) -> list[tuple]:
    """Compare results by their values, ignoring column names and row order,
    and rounding numbers, so `pct` vs `delay_rate` or 23.40 vs 23.4 still match."""
    def cell(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v):
            return round(float(v), 1)
        return None if pd.isna(v) else str(v)
    rows = [tuple(sorted((cell(v) for v in row), key=repr)) for row in df.itertuples(index=False)]
    return sorted(rows, key=repr)


def results_match(got: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(got) != len(expected):
        return False
    # Extra columns (for example a flight count) are fine as long as every
    # expected column appears somewhere in the answer.
    exp_rows = _normalise(expected)
    if got.shape[1] == expected.shape[1]:
        return _normalise(got) == exp_rows
    for cols in _column_subsets(got, expected.shape[1]):
        if _normalise(got[list(cols)]) == exp_rows:
            return True
    return False


def _column_subsets(df: pd.DataFrame, k: int):
    from itertools import combinations
    return combinations(df.columns, k)


def evaluate(llm: LLM, con: duckdb.DuckDBPyConnection, path: Path = EVAL_PATH) -> dict:
    """Run every question in the eval set and compare with the hand-written answer."""
    cases = json.loads(Path(path).read_text())
    results = []
    for case in cases:
        a = ask(case["question"], llm, con)
        if case.get("expect") == "refuse":
            ok = a.refused is not None and a.table is None
        else:
            expected, _ = run_sql(con, case["sql"])
            ok = a.table is not None and results_match(a.table, expected)
        results.append({"question": case["question"], "ok": ok, "sql": a.sql, "refused": a.refused})
    correct = sum(r["ok"] for r in results)
    return {"correct": correct, "total": len(results),
            "accuracy": round(correct / len(results), 3), "cases": results}


def format_answer(a: Answer) -> str:
    if a.refused:
        return f"Could not answer: {a.refused}" + (f"\n\nSQL tried:\n{a.sql}" if a.sql else "")
    table = a.table.to_string(index=False) if len(a.table) else "(no rows)"
    notes = "".join(f"\n({n})" for n in a.notes)
    return f"{table}{notes}\n\nSQL:\n{a.sql}"
