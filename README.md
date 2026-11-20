# Solar Power Prediction Application

An end-to-end machine-learning pipeline that predicts hourly solar power output for **Pune, Maharashtra (18.52° N, 73.86° E)**. It cleans the provided 60-day dataset, engineers time features, trains and evaluates regression models, fetches a live 48-hour weather forecast, and visualizes historical data together with the predicted output.

## Final visualization

![Historical data and 48-hour forecast](outputs/solar_forecast_chart.png)

*Last 7 days of historical `solar_output` followed by the 48-hour forecast. Historical dates are shifted for display only (see design decisions).* An interactive Plotly version is saved as `outputs/solar_forecast_interactive.html`.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt     # install dependencies
python train.py                     # generate data, train, evaluate, save models/solar_model.joblib
```

Then run the notebook (full walkthrough with charts and tables) or the apps:

```bash
jupyter notebook                    # open Solar_Power_Prediction.ipynb and Run All Cells
streamlit run app.py                # web app      -> http://localhost:8501
uvicorn api:app --reload            # REST API     -> http://127.0.0.1:8000/docs
python -m pytest                    # unit tests
```

**API key:** none required. See the next section.

## Weather API

I used the **[Open-Meteo Forecast API](https://open-meteo.com/)** (`https://api.open-meteo.com/v1/forecast`). It is free for non-commercial use and needs **no sign-up and no API key**, so there is nothing to configure and no secret to commit. (If a keyed API is used instead, put the key in a `.env` file, which is already in `.gitignore`.)

- **Fields used:** `temperature_2m`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`, `precipitation`
- **Settings:** hourly data, `timezone=Asia/Kolkata`, `wind_speed_unit=ms`; the next 48 hours from the current hour are kept
- **API errors / rate limits:** up to 3 attempts with increasing wait; HTTP 429 is handled by waiting (using `Retry-After` when sent) and retrying
- **Missing fields:** single missing hours are interpolated; a field missing entirely is filled with its training-data median
- **API downtime:** the last successful forecast is saved to `last_forecast.csv` and used as a fallback (the app and API report the source as "saved"); if nothing is saved, a clear error is returned (HTTP 503 from the REST API)
- **Caching:** live forecasts are cached for 10 minutes to limit API calls

## Model performance (validation set)

The data is split **chronologically**: the first 50 days (1,200 rows, 1 Jan – 19 Feb 2024) are used for training and the last 10 days (240 rows, 20 – 29 Feb 2024) for validation. Hyperparameters are tuned with `GridSearchCV` + `TimeSeriesSplit` **inside the training days only**, and the model with the lowest cross-validation RMSE is selected.

| Model | CV RMSE (train) | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| LightGBM (selected) | 22.74 | 15.32 | 20.26 | 0.9852 |
| XGBoost | 23.02 | 15.36 | 20.98 | 0.9842 |
| Random Forest | 25.84 | 15.07 | 20.71 | 0.9846 |

**Selected model: LightGBM** (`n_estimators=200`, `num_leaves=15`). `solar_output` is in the assignment's arbitrary units (~kW scale), so MAE and RMSE use the same units. The three models perform very similarly; results may vary slightly with library versions.

## Sample output: next 24 hours

Live Open-Meteo forecast for Pune, generated on 19 Nov 2026 (`outputs/sample_24h_predictions.csv`).

| Date and time | Predicted solar output |
|---|---:|
| 19 Nov 2026 16:00 | 521.73 |
| 19 Nov 2026 17:00 | 226.94 |
| 19 Nov 2026 18:00 | 21.44 |
| 19 Nov 2026 19:00 | 19.39 |
| 19 Nov 2026 20:00 | 11.60 |
| 19 Nov 2026 21:00 | 17.01 |
| 19 Nov 2026 22:00 | 12.82 |
| 19 Nov 2026 23:00 | 9.87 |
| 20 Nov 2026 00:00 | 4.39 |
| 20 Nov 2026 01:00 | 13.36 |
| 20 Nov 2026 02:00 | 16.38 |
| 20 Nov 2026 03:00 | 9.41 |
| 20 Nov 2026 04:00 | 9.51 |
| 20 Nov 2026 05:00 | 10.58 |
| 20 Nov 2026 06:00 | 9.64 |
| 20 Nov 2026 07:00 | 102.20 |
| 20 Nov 2026 08:00 | 413.94 |
| 20 Nov 2026 09:00 | 578.03 |
| 20 Nov 2026 10:00 | 669.31 |
| 20 Nov 2026 11:00 | 678.94 |
| 20 Nov 2026 12:00 | 675.41 |
| 20 Nov 2026 13:00 | 675.06 |
| 20 Nov 2026 14:00 | 608.11 |
| 20 Nov 2026 15:00 | 587.61 |

## Design decisions and assumptions

1. **Dataset:** generated once with the exact script from the brief (`np.random.seed(42)`), giving 1,440 hourly rows.
2. **Cleaning:** timestamps are parsed and sorted, duplicate hours removed, a complete hourly timeline rebuilt, gaps filled by linear interpolation, and values clipped to valid ranges (e.g. humidity 0–100).
3. **Features:** hour (0–23), `sin`/`cos` of hour (so 23:00 and 00:00 are neighbours), day of week and weekend flag. One shared function, `add_time_features()`, builds them for both historical and forecast data, so training and inference use identical columns.
4. **Chronological split:** no shuffling, so the model is never trained on data from after the validation period.
5. **Model choice:** Random Forest, XGBoost and LightGBM are compared; the final choice is made on cross-validation RMSE, not on the validation scores.
6. **No negative output:** predictions are clipped with `np.clip(pred, 0, None)`.
7. **Timezone:** dataset hours and API data are both treated as local time (`Asia/Kolkata`); only forecast hours from the current hour onward are used.
8. **Chart:** the dataset covers Jan–Feb 2024 while the forecast is for the current date. To draw one continuous line, the history's dates are shifted for display only; data and model are unchanged.
9. **Synthetic data:** scores show how well the model learns the generated data, not real solar-plant accuracy. In the brief's generator, night-time output is `max(0, noise)`, so night predictions are small positive values rather than exactly 0.

## Project structure

```text
Solar_Power_Prediction.ipynb   # step-by-step workflow: data, features, model, forecast, charts
solar_core.py                  # shared code: data, features, weather API, prediction
train.py                       # training script: split, tune, evaluate, save model
app.py                         # Streamlit web app
api.py                         # FastAPI REST API
tests/                         # pytest unit tests
requirements.txt
models/solar_model.joblib      # saved model (created by train.py)
outputs/                       # charts, prediction tables, metrics
```

## Bonus features

- Comparison of three models (Random Forest, XGBoost, LightGBM)
- Interactive Plotly chart (notebook and web app)
- Small REST API with FastAPI: `/health`, `/model`, `/forecast?hours=24`, `/predict`
- Cached, rate-limit-aware API calls with retries and a saved-forecast fallback
- 36 unit tests (pytest) covering feature engineering, the chronological split, cleaning, API handling and the REST endpoints; the weather API is mocked, so no internet is needed
