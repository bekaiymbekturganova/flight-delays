"""Command line entry point:  flightdelays <step>"""
from __future__ import annotations

import argparse
import json

from . import clean as clean_step
from . import download as download_step
from . import features as features_step
from . import model as model_step
from . import report as report_step
from . import weather as weather_step


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flightdelays", description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("download", "all"):
        p = sub.add_parser(name)
        p.add_argument("--start", required=True, help="first month, YYYY-MM")
        p.add_argument("--end", required=True, help="last month, YYYY-MM")
    for name in ("clean", "weather", "features", "train", "report"):
        sub.add_parser(name)
    p = sub.add_parser("ask", help="ask the flight data a question in plain English")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the answer as JSON")
    p = sub.add_parser("ask-eval", help="score the question-to-SQL assistant on the eval set")
    p.add_argument("--out", help="also save every question, SQL and verdict to this JSON file")
    args = parser.parse_args(argv)

    if args.step in ("ask", "ask-eval"):
        from . import ask as ask_step
        con = ask_step.connect()
        llm = ask_step.anthropic_llm()
        if args.step == "ask":
            answer = ask_step.ask(args.question, llm, con)
            print(json.dumps(answer.as_dict(), indent=2) if args.json else ask_step.format_answer(answer))
        else:
            result = ask_step.evaluate(llm, con)
            for case in result["cases"]:
                print(("ok   " if case["ok"] else "MISS ") + case["question"])
            print(f"\n{result['correct']}/{result['total']} correct ({result['accuracy']:.0%})")
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(result, f, indent=2)
        return

    if args.step in ("download", "all"):
        download_step.download(args.start, args.end)
    if args.step in ("clean", "all"):
        print(f"clean: kept {clean_step.clean():,} flights", flush=True)
    if args.step in ("weather", "all"):
        print(f"weather: {weather_step.download_weather():,} airport-days")
    if args.step in ("features", "all"):
        print(f"features: {features_step.build_features():,} rows")
    if args.step in ("train", "all"):
        metrics = model_step.train()
        print(json.dumps(metrics["test_metrics"], indent=2))
    if args.step in ("report", "all"):
        print(f"report: {report_step.write_report()}")


if __name__ == "__main__":
    main()
