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
    args = parser.parse_args(argv)

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
