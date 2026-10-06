"""Build the results site: an interactive page plus the data file it reads."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import ARTIFACTS_DIR, METRICS_PATH
from .model import CUBE_PATH

DASHBOARD = Path(__file__).with_name("dashboard.html")


def write_report(artifacts_dir: Path = ARTIFACTS_DIR, out_dir: Path = Path("site")) -> Path:
    """Copy the dashboard and write site/data.json from the training outputs."""
    metrics = json.loads((artifacts_dir / METRICS_PATH.name).read_text())
    cube = json.loads((artifacts_dir / CUBE_PATH.name).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(
        json.dumps({"metrics": metrics, "cube": cube}, separators=(",", ":"))
    )
    shutil.copyfile(DASHBOARD, out_dir / "index.html")
    return out_dir / "index.html"
