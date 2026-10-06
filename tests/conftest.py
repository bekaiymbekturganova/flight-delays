import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from sample import write_raw  # noqa: E402

from flightdelays.clean import clean  # noqa: E402
from flightdelays.features import build_features  # noqa: E402


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    raw = write_raw(root / "raw")
    kept = clean(root / "raw", root / "flights.parquet")
    build_features(root / "flights.parquet", root / "features.parquet")
    return {"root": root, "raw": raw, "kept": kept}
