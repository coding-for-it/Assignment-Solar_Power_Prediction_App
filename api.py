"""
api.py - small REST API for the solar power model (FastAPI).

Run:   uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import solar_core as core

CACHE_SECONDS = 600                          # reuse a live forecast for 10 minutes
state = {"bundle": None, "load_error": None}
forecast_cache = {"time": 0.0, "weather": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the server starts."""
    state["bundle"], state["load_error"] = None, None
    try:
        state["bundle"] = core.load_bundle()
    except Exception as error:               # server still starts; /health explains the problem
        state["load_error"] = str(error)
    yield


app = FastAPI(
    title="Solar Power Prediction API",
    description="Hourly solar output predictions for Pune, India using a live weather forecast.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Response / request models
# ----------------------------------------------------------------------------
class HourPrediction(BaseModel):
    timestamp: str
    predicted_solar_output: float
    temperature: float
    humidity: float
    cloud_cover: float
    wind_speed: float
    precipitation: float


class ForecastResponse(BaseModel):
    location: str
    source: str = Field(description="'live' = fresh Open-Meteo data, 'saved' = last saved forecast")
    hours: int
    unit_note: str
    predictions: List[HourPrediction]


class WeatherInput(BaseModel):
    timestamp: str = Field(examples=["2026-11-20T12:00"])
    temperature: float = Field(ge=-50, le=60, description="deg C")
    humidity: float = Field(ge=0, le=100, description="%")
    cloud_cover: float = Field(ge=0, le=100, description="%")
    wind_speed: float = Field(ge=0, le=100, description="m/s")
    precipitation: float = Field(ge=0, le=500, description="mm")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def require_model():
    if state["bundle"] is None:
        raise HTTPException(503, f"Model not loaded. Run `python train.py` first. ({state['load_error']})")
    return state["bundle"]


def to_predictions(result):
    return [HourPrediction(
        timestamp=row.timestamp.strftime("%Y-%m-%dT%H:%M"),
        predicted_solar_output=float(row.predicted_solar_output),
        temperature=round(float(row.temperature), 2),
        humidity=round(float(row.humidity), 2),
        cloud_cover=round(float(row.cloud_cover), 2),
        wind_speed=round(float(row.wind_speed), 2),
        precipitation=round(float(row.precipitation), 2),
    ) for row in result.itertuples()]


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Is the server up and the model loaded?"""
    return {"status": "ok" if state["bundle"] is not None else "model_not_loaded",
            "model_loaded": state["bundle"] is not None,
            "detail": state["load_error"]}


@app.get("/model")
def model_info():
    """Selected model, feature list and validation metrics (last 10 days)."""
    bundle = require_model()
    metrics = json.loads(core.METRICS_FILE.read_text()) if core.METRICS_FILE.exists() else None
    return {"model": bundle["model_name"], "features": bundle["features"],
            "validation_metrics": metrics}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(hours: int = Query(24, ge=1, le=core.FORECAST_HOURS,
                                description="How many future hours to predict (1-48)")):
    """Fetch the weather forecast for Pune and predict solar output for the next N hours."""
    bundle = require_model()

    fresh = forecast_cache["weather"] is not None and time.time() - forecast_cache["time"] < CACHE_SECONDS
    if fresh:
        weather, source = forecast_cache["weather"], "live"
    else:
        try:
            weather, source = core.get_forecast(bundle["fill_values"])
        except RuntimeError as error:
            raise HTTPException(503, str(error))
        if source == "live":
            forecast_cache.update(time=time.time(), weather=weather)

    result = core.predict(bundle, weather.head(hours))
    return ForecastResponse(
        location="Pune, Maharashtra, India (18.52, 73.86)", source=source, hours=len(result),
        unit_note="solar_output is in arbitrary units on a ~kW scale (as defined in the assignment)",
        predictions=to_predictions(result))


@app.post("/predict", response_model=List[HourPrediction])
def predict_custom(rows: List[WeatherInput]):
    """Predict solar output for weather values you provide (no weather API call)."""
    bundle = require_model()
    if not rows:
        raise HTTPException(422, "Send at least one weather row.")
    import pandas as pd
    weather = pd.DataFrame([r.model_dump() for r in rows])
    try:
        weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    except (ValueError, TypeError):
        raise HTTPException(422, "timestamp must look like 2026-11-20T12:00")
    return to_predictions(core.predict(bundle, weather))