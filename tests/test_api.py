"""Tests for the FastAPI endpoints (no internet needed: the weather call is faked)."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import solar_core as core  # noqa: E402


def fake_weather(hours=48):
    """48 hours of clear-sky weather starting at 10:00."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-11-20 10:00", periods=hours, freq="h"),
        "temperature": 28.0, "humidity": 40.0, "cloud_cover": 5.0,
        "wind_speed": 2.0, "precipitation": 0.0})


@pytest.fixture
def client(bundle):                      # `bundle` fixture skips these tests if no model exists
    api.forecast_cache.update(time=0.0, weather=None)      # start every test with an empty cache
    with TestClient(api.app) as test_client:
        yield test_client


GOOD_ROW = {"timestamp": "2026-11-20T12:00", "temperature": 30, "humidity": 50,
            "cloud_cover": 10, "wind_speed": 3, "precipitation": 0}


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_endpoint_lists_features(client):
    body = client.get("/model").json()
    assert body["features"] == core.FEATURES
    assert body["model"]


def test_forecast_returns_requested_hours_and_no_negatives(client, monkeypatch):
    monkeypatch.setattr(core, "get_forecast", lambda fill_values: (fake_weather(), "live"))
    response = client.get("/forecast?hours=24")
    body = response.json()

    assert response.status_code == 200
    assert body["source"] == "live" and body["hours"] == 24 and len(body["predictions"]) == 24
    assert all(p["predicted_solar_output"] >= 0 for p in body["predictions"])
    assert {"timestamp", "temperature", "humidity", "cloud_cover", "wind_speed",
            "precipitation"} <= set(body["predictions"][0])


def test_forecast_reports_saved_source_when_api_is_down(client, monkeypatch):
    monkeypatch.setattr(core, "get_forecast", lambda fill_values: (fake_weather(), "saved"))
    assert client.get("/forecast?hours=5").json()["source"] == "saved"


def test_forecast_returns_503_when_no_weather_available(client, monkeypatch):
    def unavailable(fill_values):
        raise RuntimeError("Weather API unavailable and no saved forecast")

    monkeypatch.setattr(core, "get_forecast", unavailable)
    assert client.get("/forecast").status_code == 503


@pytest.mark.parametrize("hours", [0, 49, -3])
def test_forecast_rejects_invalid_hours(client, hours):
    assert client.get(f"/forecast?hours={hours}").status_code == 422


def test_predict_endpoint_midday_is_higher_than_midnight(client):
    night = dict(GOOD_ROW, timestamp="2026-11-20T00:00", temperature=18)
    response = client.post("/predict", json=[GOOD_ROW, night])
    noon_pred, night_pred = [p["predicted_solar_output"] for p in response.json()]

    assert response.status_code == 200
    assert noon_pred > night_pred >= 0


@pytest.mark.parametrize("bad_row", [
    dict(GOOD_ROW, humidity=150),            # out of range
    dict(GOOD_ROW, cloud_cover=-1),          # out of range
    dict(GOOD_ROW, timestamp="nonsense"),    # not a date
])
def test_predict_endpoint_rejects_bad_input(client, bad_row):
    assert client.post("/predict", json=[bad_row]).status_code == 422


def test_predict_endpoint_rejects_empty_list(client):
    assert client.post("/predict", json=[]).status_code == 422
