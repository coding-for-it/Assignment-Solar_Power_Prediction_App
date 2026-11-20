"""Unit tests for solar_core.py: data, features, split, prediction and weather-API handling."""
import numpy as np
import pandas as pd
import pytest
import requests

import solar_core as core


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def future_hourly(hours=72):
    """Fake Open-Meteo 'hourly' block that starts at midnight today (India time)."""
    start = pd.Timestamp.now(tz=core.TIMEZONE).normalize().tz_localize(None)
    times = pd.date_range(start, periods=hours, freq="h")
    return {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [25.0] * hours,
        "relative_humidity_2m": [50.0] * hours,
        "cloud_cover": [20.0] * hours,
        "wind_speed_10m": [3.0] * hours,
        "precipitation": [0.0] * hours,
    }


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummyModel:
    """Pretends to be a model that predicts a negative number."""
    def predict(self, X):
        return np.full(len(X), -50.0)


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
def test_generate_dataset_matches_assignment(tmp_path):
    path = tmp_path / "data.csv"
    data = core.generate_dataset(path)
    assert data.shape == (1440, 7)
    assert list(data.columns) == ["timestamp", *core.WEATHER_COLUMNS, core.TARGET]
    assert data["timestamp"].iloc[0] == "2024-01-01 00:00:00"
    assert data["timestamp"].iloc[-1] == "2024-02-29 23:00:00"
    assert (data[core.TARGET] >= 0).all()


def test_generate_dataset_is_reproducible(tmp_path):
    first = core.generate_dataset(tmp_path / "a.csv")
    second = core.generate_dataset(tmp_path / "b.csv")
    pd.testing.assert_frame_equal(first, second)


# ----------------------------------------------------------------------------
# cleaning
# ----------------------------------------------------------------------------
def test_load_and_clean_fixes_gaps_duplicates_and_outliers(tmp_path):
    csv = tmp_path / "messy.csv"
    csv.write_text(
        "timestamp,temperature,humidity,cloud_cover,wind_speed,precipitation,solar_output\n"
        "2024-01-01 00:00:00,20,50,10,3,0,0\n"
        "2024-01-01 01:00:00,22,150,10,3,0,0\n"          # humidity outlier
        "2024-01-01 01:00:00,22,150,10,3,0,0\n"          # duplicate hour
        "2024-01-01 03:00:00,26,50,10,3,0,300\n"         # 02:00 row missing
        "2024-01-01 04:00:00,,50,10,3,0,400\n"           # missing temperature
    )
    df = core.load_and_clean(csv)

    assert len(df) == 5                                   # complete hourly timeline 00:00-04:00
    assert not df["timestamp"].duplicated().any()
    assert df.drop(columns="timestamp").isna().sum().sum() == 0
    assert df["humidity"].max() <= 100                    # outlier clipped
    assert df.loc[2, "temperature"] == pytest.approx(24.0)  # interpolated between 22 and 26


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def test_time_features_values():
    data = pd.DataFrame({"timestamp": ["2024-01-01 00:00:00",    # Monday, midnight
                                       "2024-01-01 06:00:00",    # Monday, 06:00
                                       "2024-01-06 12:00:00"]})  # Saturday, noon
    out = core.add_time_features(data)

    assert out["hour"].tolist() == [0, 6, 12]
    assert out["hour_sin"].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert out["hour_cos"].tolist() == pytest.approx([1.0, 0.0, -1.0], abs=1e-9)
    assert out["day_of_week"].tolist() == [0, 0, 5]
    assert out["is_weekend"].tolist() == [0, 0, 1]


def test_cyclic_encoding_wraps_around_midnight():
    out = core.add_time_features(pd.DataFrame(
        {"timestamp": ["2024-01-01 23:00:00", "2024-01-02 00:00:00", "2024-01-02 12:00:00"]}))
    xy = out[["hour_sin", "hour_cos"]].to_numpy()
    near = np.linalg.norm(xy[0] - xy[1])     # 23:00 vs 00:00
    far = np.linalg.norm(xy[1] - xy[2])      # 00:00 vs 12:00
    assert near < far / 5


def test_add_time_features_does_not_change_input():
    data = pd.DataFrame({"timestamp": ["2024-01-01 00:00:00"]})
    core.add_time_features(data)
    assert list(data.columns) == ["timestamp"]


# ----------------------------------------------------------------------------
# chronological split
# ----------------------------------------------------------------------------
def test_chronological_split_50_train_10_validation(clean_dataset):
    train, valid = clean_dataset.iloc[:core.TRAIN_ROWS], clean_dataset.iloc[core.TRAIN_ROWS:]
    assert (len(train), len(valid)) == (1200, 240)
    assert train["timestamp"].max() < valid["timestamp"].min()          # no overlap, time order kept
    assert train["timestamp"].max() == pd.Timestamp("2024-02-19 23:00")
    assert valid["timestamp"].min() == pd.Timestamp("2024-02-20 00:00")


# ----------------------------------------------------------------------------
# prediction
# ----------------------------------------------------------------------------
def test_predict_never_returns_negative_values():
    bundle = {"model": DummyModel(), "features": core.FEATURES}
    weather = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
                            **{c: 10.0 for c in core.WEATHER_COLUMNS}})
    result = core.predict(bundle, weather)
    assert (result["predicted_solar_output"] == 0).all()


def test_predict_raises_if_model_feature_is_missing():
    bundle = {"model": DummyModel(), "features": core.FEATURES + ["not_a_real_feature"]}
    weather = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
                            **{c: 10.0 for c in core.WEATHER_COLUMNS}})
    with pytest.raises(ValueError):
        core.predict(bundle, weather)


def test_real_model_features_match_schema(bundle):
    assert bundle["features"] == core.FEATURES
    assert set(core.WEATHER_COLUMNS) <= set(bundle["fill_values"])


def test_real_model_day_is_higher_than_night(bundle):
    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-06-01 12:00", "2026-06-01 00:00"]),
        "temperature": [30.0, 18.0], "humidity": [50.0, 60.0], "cloud_cover": [10.0, 10.0],
        "wind_speed": [3.0, 3.0], "precipitation": [0.0, 0.0]})
    noon, midnight = core.predict(bundle, weather)["predicted_solar_output"]
    assert noon > 300
    assert 0 <= midnight < 50


def test_real_model_quality_on_validation_days(bundle, clean_dataset):
    valid = clean_dataset.iloc[core.TRAIN_ROWS:]
    metrics = core.evaluate(bundle["model"], valid[bundle["features"]], valid[core.TARGET])
    assert metrics["R2"] > 0.95
    assert metrics["MAE"] < 30


# ----------------------------------------------------------------------------
# weather forecast -> DataFrame
# ----------------------------------------------------------------------------
FILL = {c: 1.5 for c in core.WEATHER_COLUMNS}


def test_build_forecast_df_matches_model_schema_and_is_future_only():
    out = core.build_forecast_df(future_hourly(), FILL)
    now = pd.Timestamp.now(tz=core.TIMEZONE).floor("h").tz_localize(None)

    assert list(out.columns) == ["timestamp", *core.WEATHER_COLUMNS]
    assert 24 <= len(out) <= core.FORECAST_HOURS
    assert out["timestamp"].min() >= now                      # past hours of today removed
    assert out["timestamp"].is_monotonic_increasing
    assert out[core.WEATHER_COLUMNS].isna().sum().sum() == 0


def test_build_forecast_df_interpolates_single_missing_values():
    hourly = future_hourly()
    hourly["temperature_2m"][30] = None
    out = core.build_forecast_df(hourly, FILL)
    assert out[core.WEATHER_COLUMNS].isna().sum().sum() == 0
    assert (out["temperature"] == 25.0).all()                 # neighbours are 25 -> filled with 25


def test_build_forecast_df_fills_a_missing_field_with_training_median():
    hourly = future_hourly()
    del hourly["wind_speed_10m"]                              # API did not send wind at all
    out = core.build_forecast_df(hourly, FILL)
    assert (out["wind_speed"] == 1.5).all()


def test_build_forecast_df_clips_out_of_range_values():
    hourly = future_hourly()
    hourly["relative_humidity_2m"] = [150.0] * 72
    hourly["cloud_cover"] = [-5.0] * 72
    out = core.build_forecast_df(hourly, FILL)
    assert out["humidity"].max() == 100
    assert out["cloud_cover"].min() == 0


def test_build_forecast_df_raises_when_no_future_rows():
    hourly = future_hourly()
    hourly["time"] = ["2020-01-01T00:00"] * 72               # all in the past
    with pytest.raises(ValueError):
        core.build_forecast_df(hourly, FILL)


# ----------------------------------------------------------------------------
# weather API: retries, rate limit, fallback
# ----------------------------------------------------------------------------
def test_fetch_hourly_retries_after_network_error(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary failure")
        return FakeResponse(payload={"hourly": {"time": []}})

    monkeypatch.setattr(core.requests, "get", fake_get)
    assert core.fetch_hourly() == {"time": []}
    assert len(calls) == 2


def test_fetch_hourly_waits_and_retries_on_rate_limit(monkeypatch):
    waits = []
    monkeypatch.setattr(core.time, "sleep", waits.append)          # record how long we wait
    responses = iter([FakeResponse(429, headers={"Retry-After": "7"}),
                      FakeResponse(payload={"hourly": {"time": ["x"]}})])
    monkeypatch.setattr(core.requests, "get", lambda *a, **k: next(responses))

    assert core.fetch_hourly() == {"time": ["x"]}
    assert waits == [7]                                            # waited the time the server asked for


def test_fetch_hourly_gives_up_after_all_retries(monkeypatch):
    def always_fail(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(core.requests, "get", always_fail)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        core.fetch_hourly(retries=3)


def test_get_forecast_falls_back_to_saved_copy(monkeypatch, tmp_path):
    saved = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
                          **{c: 1.0 for c in core.WEATHER_COLUMNS}})
    cache = tmp_path / "last_forecast.csv"
    saved.to_csv(cache, index=False)

    def api_down(fill_values):
        raise RuntimeError("API down")

    monkeypatch.setattr(core, "CACHE_FILE", cache)
    monkeypatch.setattr(core, "get_live_forecast", api_down)

    forecast, source = core.get_forecast(FILL)
    assert source == "saved"
    assert len(forecast) == 3


def test_get_forecast_raises_when_api_down_and_nothing_saved(monkeypatch, tmp_path):
    def api_down(fill_values):
        raise RuntimeError("API down")

    monkeypatch.setattr(core, "CACHE_FILE", tmp_path / "missing.csv")
    monkeypatch.setattr(core, "get_live_forecast", api_down)
    with pytest.raises(RuntimeError, match="no saved forecast"):
        core.get_forecast(FILL)


def test_get_forecast_reports_live_source(monkeypatch):
    fake = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="h")})
    monkeypatch.setattr(core, "get_live_forecast", lambda fill_values: fake)
    forecast, source = core.get_forecast(FILL)
    assert source == "live" and forecast is fake
