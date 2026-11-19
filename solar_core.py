"""
solar_core.py - shared code used by train.py, api.py and app.py.

Keeping the data, feature and weather-API logic in one place means the training
script, the REST API and the web app always use exactly the same steps.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "solar_weather_dataset.csv"
MODEL_FILE = BASE_DIR / "models" / "solar_model.joblib"
CACHE_FILE = BASE_DIR / "last_forecast.csv"          # last good API response
METRICS_FILE = BASE_DIR / "outputs" / "model_metrics.json"

TARGET = "solar_output"
WEATHER_COLUMNS = ["temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]
FEATURES = WEATHER_COLUMNS + ["hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend"]
TRAIN_ROWS = 50 * 24            # first 50 days = train, last 10 days = validation
FORECAST_HOURS = 48

API_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 18.52, 73.86       # Pune
TIMEZONE = "Asia/Kolkata"
API_TO_FEATURE = {                       # API field name -> our column name
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "cloud_cover": "cloud_cover",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def generate_dataset(path=DATA_FILE):
    """Exact dataset script from the assignment (seed 42): 60 days x 24 hours."""
    np.random.seed(42)
    rows = []
    start = datetime(2024, 1, 1)
    for i in range(24 * 60):
        ts = start + timedelta(hours=i)
        hour = ts.hour
        temperature = 18 + 10 * np.sin((hour / 24) * 2 * np.pi) + np.random.normal(0, 1)
        humidity = np.clip(60 + np.random.normal(0, 10), 20, 100)
        cloud_cover = np.clip(np.random.normal(40, 25), 0, 100)
        wind_speed = np.clip(np.random.normal(5, 2), 0, 15)
        precipitation = max(0, np.random.normal(0.2, 0.5))
        daylight_factor = max(0, np.sin((hour - 6) / 12 * np.pi))
        solar_output = max(0, daylight_factor * (1 - cloud_cover / 120) * (temperature / 30) * 1000)
        solar_output = max(0, solar_output + np.random.normal(0, 20))
        rows.append([ts.strftime("%Y-%m-%d %H:%M:%S"), temperature, humidity,
                     cloud_cover, wind_speed, precipitation, solar_output])
    data = pd.DataFrame(rows, columns=["timestamp", *WEATHER_COLUMNS, TARGET])
    data.to_csv(path, index=False)
    return data


def load_and_clean(path=DATA_FILE):
    """Parse timestamps, remove duplicates, fill gaps by interpolation, clip to valid ranges."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))   # complete hourly timeline
    df.index.name = "timestamp"
    df = df.apply(pd.to_numeric, errors="coerce").interpolate(method="linear", limit_direction="both")

    df["humidity"] = df["humidity"].clip(0, 100)
    df["cloud_cover"] = df["cloud_cover"].clip(0, 100)
    for column in ["wind_speed", "precipitation", TARGET]:
        df[column] = df[column].clip(lower=0)
    return df.reset_index()


def add_time_features(data):
    """hour, sin/cos of hour, day of week, weekend flag (same for history and forecast)."""
    out = data.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_bundle(path=MODEL_FILE):
    """Saved dict: model, model_name, features, fill_values."""
    return joblib.load(path)


def evaluate(model, X, y):
    """MAE, RMSE, R2 (predictions clipped at 0)."""
    pred = np.clip(model.predict(X), 0, None)
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}


def predict(bundle, weather):
    """Weather DataFrame -> DataFrame with a predicted_solar_output column (never negative)."""
    data = add_time_features(weather)
    missing = [c for c in bundle["features"] if c not in data.columns]
    if missing:
        raise ValueError(f"Forecast is missing model features: {missing}")
    data["predicted_solar_output"] = np.clip(
        bundle["model"].predict(data[bundle["features"]]), 0, None).round(2)
    return data


# ----------------------------------------------------------------------------
# Weather API (Open-Meteo)
# ----------------------------------------------------------------------------
def fetch_hourly(retries=3):
    """Call Open-Meteo. Retries on errors and waits on rate limit (HTTP 429)."""
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": ",".join(API_TO_FEATURE),
        "wind_speed_unit": "ms", "forecast_days": 3, "timezone": TIMEZONE,
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else 2 ** attempt
                last_error = "HTTP 429 (rate limit)"
                time.sleep(min(wait, 15))
                continue
            response.raise_for_status()
            return response.json()["hourly"]
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Weather API failed after {retries} attempts: {last_error}")


def build_forecast_df(hourly, fill_values, hours=FORECAST_HOURS):
    """API response -> DataFrame with our column names; fills gaps; keeps future hours only."""
    out = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    for api_name, column in API_TO_FEATURE.items():
        values = hourly.get(api_name)
        out[column] = pd.to_numeric(pd.Series(values), errors="coerce") if values else np.nan

    out = out.sort_values("timestamp").set_index("timestamp")
    out[WEATHER_COLUMNS] = out[WEATHER_COLUMNS].interpolate(method="time", limit_direction="both")
    for column in WEATHER_COLUMNS:                    # whole field missing -> training median
        out[column] = out[column].fillna(fill_values.get(column, 0))
    out = out.reset_index()

    out["humidity"] = out["humidity"].clip(0, 100)
    out["cloud_cover"] = out["cloud_cover"].clip(0, 100)

    now = pd.Timestamp.now(tz=TIMEZONE).floor("h").tz_localize(None)
    out = out[out["timestamp"] >= now].head(hours)
    if out.empty:
        raise ValueError("Open-Meteo returned no upcoming hourly rows.")
    return out.reset_index(drop=True)


def get_live_forecast(fill_values):
    """Fetch + build the forecast and keep a copy in case the API is down next time."""
    forecast = build_forecast_df(fetch_hourly(), fill_values)
    CACHE_FILE.parent.mkdir(exist_ok=True)
    forecast.to_csv(CACHE_FILE, index=False)
    return forecast


def load_saved_forecast():
    """Last saved forecast, or None."""
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
    return None


def get_forecast(fill_values):
    """Return (forecast, source). source is 'live' or 'saved'; raises if neither works."""
    try:
        return get_live_forecast(fill_values), "live"
    except Exception as error:
        saved = load_saved_forecast()
        if saved is None:
            raise RuntimeError(f"Weather API unavailable and no saved forecast: {error}")
        return saved, "saved"