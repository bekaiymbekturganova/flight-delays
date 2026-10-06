"""Write a small static HTML report from artifacts/metrics.json."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .config import ARTIFACTS_DIR, METRICS_PATH

LABELS = {
    "route_history": "Route's 28-day delay rate",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting (LightGBM)",
}

STYLE = """
:root{--bg:#f5f6f9;--ink:#0e1a33;--muted:#465069;--line:#d3d8e3;--accent:#7b1e36;--bar:#0e1a33}
@media (prefers-color-scheme:dark){:root{--bg:#0a1122;--ink:#eceef4;--muted:#9aa5bd;--line:#26324d;--accent:#e99cac;--bar:#9fb4e6}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 20px 64px}
main{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1{font-size:2.2rem;line-height:1.1;margin:0}h2{font-size:1.25rem;margin:0 0 12px}
p{margin:0;max-width:65ch}.muted{color:var(--muted)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{font-size:.85rem;color:var(--muted);font-weight:600}
tr.best td{font-weight:700;color:var(--accent)}
.bar{display:inline-block;height:10px;background:var(--bar);border-radius:2px;vertical-align:middle;margin-right:8px}
a{color:var(--accent)}
"""


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(metrics: dict) -> str:
    data = metrics["data"]
    results = metrics["test_metrics"]
    best = max(results, key=lambda k: results[k]["pr_auc"])
    rows = "".join(
        f"<tr class='{'best' if k == best else ''}'><td>{html.escape(LABELS.get(k, k))}</td>"
        f"<td>{v['pr_auc']:.3f}</td><td>{v['roc_auc']:.3f}</td><td>{v['brier']:.4f}</td></tr>"
        for k, v in results.items()
    )
    calib = "".join(
        f"<tr><td>{i + 1}</td><td>{_pct(r['predicted'])}</td><td>{_pct(r['observed'])}</td><td>{r['flights']:,}</td></tr>"
        for i, r in enumerate(metrics["calibration"])
    )
    peak = max((r["observed"] for r in metrics["by_hour"]), default=1) or 1
    hours = "".join(
        f"<tr><td>{int(r['dep_hour']):02d}:00</td>"
        f"<td style='text-align:left'><span class='bar' style='width:{160 * r['observed'] / peak:.0f}px'></span>{_pct(r['observed'])}</td>"
        f"<td>{_pct(r['predicted'])}</td><td>{int(r['flights']):,}</td></tr>"
        for r in metrics["by_hour"]
    )
    top = metrics["importance"][:8]
    imp = "".join(
        f"<tr><td>{html.escape(r['feature'])}</td>"
        f"<td style='text-align:left'><span class='bar' style='width:{240 * r['share'] / (top[0]['share'] or 1):.0f}px'></span>{_pct(r['share'])}</td></tr>"
        for r in top
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flight delay risk: results</title><style>{STYLE}</style></head><body><main>
<header>
<h1>Will this flight leave late?</h1>
<p class="muted">Day-ahead prediction of departure delays of 15+ minutes, from public US Bureau of Transportation Statistics data.</p>
</header>
<section><h2>Data</h2>
<p>{data['first_day']} to {data['last_day']}. Trained on {data['train']['flights']:,} flights
({data['train']['from']} to {data['train']['to']}), tuned on {data['valid']['flights']:,}, and tested on
{data['test']['flights']:,} later flights ({data['test']['from']} to {data['test']['to']}).
{_pct(data['test_delay_rate'])} of test flights left 15+ minutes late.</p></section>
<section><h2>Results on the test period</h2>
<div class="scroll"><table><thead><tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Brier</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="muted">PR-AUC of a random guess equals the delay rate, {data['test_delay_rate']:.3f}. Lower Brier is better.</p></section>
<section><h2>Are the probabilities honest?</h2>
<p class="muted">Test flights sorted by predicted risk and cut into equal groups. A calibrated model has matching columns.</p>
<div class="scroll"><table><thead><tr><th>Group</th><th>Predicted</th><th>Observed</th><th>Flights</th></tr></thead><tbody>{calib}</tbody></table></div></section>
<section><h2>Delay rate by scheduled departure hour</h2>
<div class="scroll"><table><thead><tr><th>Hour</th><th style="text-align:left">Observed</th><th>Predicted</th><th>Flights</th></tr></thead><tbody>{hours}</tbody></table></div></section>
<section><h2>What the model relies on</h2>
<div class="scroll"><table><thead><tr><th>Feature</th><th style="text-align:left">Share of total gain</th></tr></thead><tbody>{imp}</tbody></table></div></section>
</main></body></html>"""


def write_report(metrics_path: Path = ARTIFACTS_DIR / METRICS_PATH.name, out_dir: Path = Path("site")) -> Path:
    metrics = json.loads(Path(metrics_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(render(metrics))
    return out
