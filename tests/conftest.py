"""Make the project root importable (so `import solar_core` works) and share fixtures."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solar_core as core  # noqa: E402


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry loops call time.sleep; skip the waiting so tests run instantly."""
    monkeypatch.setattr(core.time, "sleep", lambda seconds: None)


@pytest.fixture(scope="session")
def bundle():
    """The real trained model bundle (created by `python train.py` or the notebook)."""
    if not core.MODEL_FILE.exists():
        pytest.skip("models/solar_model.joblib not found - run `python train.py` first")
    return core.load_bundle()


@pytest.fixture(scope="session")
def clean_dataset(tmp_path_factory):
    """The assignment dataset, regenerated in a temp folder, cleaned and with features."""
    path = tmp_path_factory.mktemp("data") / "dataset.csv"
    core.generate_dataset(path)
    return core.add_time_features(core.load_and_clean(path))
