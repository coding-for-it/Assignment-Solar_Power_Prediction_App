"""
Solar Power Prediction - Streamlit app (Pune, India)

Loads the model trained by train.py / the notebook, fetches the next 48 hours of
weather from Open-Meteo, predicts solar output and shows a chart + table.

Run train.py (or the notebook) first, then:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import solar_core as core

PRODUCTIVE_THRESHOLD = 100      # output above this counts as a "productive hour"

st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️", layout="wide")


# ----------------------------------------------------------------------------
# Model + history (cached so they load once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_history():
    """Load the saved model and the CSV; score the model on the last 10 days."""
    bundle = core.load_bundle()
    history = core.add_time_features(core.load_and_clean(core.DATA_FILE))

    valid = history.iloc[core.TRAIN_ROWS:]          # last 240 rows, never used for training
    metrics = core.evaluate(bundle["model"], valid[bundle["features"]], valid[core.TARGET])
    return bundle, history, metrics


@st.cache_data(ttl=600, show_spinner=False)         # cache 10 minutes -> fewer API calls
def cached_live_forecast(fill_values):
    return core.get_live_forecast(fill_values)


def get_forecast(fill_values):
    """Return (forecast, source). Falls back to the last saved forecast if the API fails."""
    try:
        return cached_live_forecast(fill_values), "live"
    except Exception as error:
        saved = core.load_saved_forecast()
        if saved is None:
            raise RuntimeError(f"Weather API unavailable and no saved forecast. ({error})")
        return saved, f"saved ({error})"


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def shift_history_for_chart(history, first_forecast_time):
    """Shift the last 7 days of history (display only) so it ends just before the forecast."""
    recent = history[["timestamp", core.TARGET]].tail(7 * 24).copy()
    shift = (first_forecast_time - pd.Timedelta(hours=1)) - recent["timestamp"].iloc[-1]
    recent["timestamp"] = recent["timestamp"] + shift
    return recent


def build_chart(history, forecast, show_history):
    fig = go.Figure()
    forecast_x = forecast["timestamp"].tolist()
    forecast_y = forecast["predicted_solar_output"].tolist()

    if show_history:
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history[core.TARGET], mode="lines",
            name="Previous 7 days (dates shifted for display)", line=dict(width=2)))
        # join the two lines so the graph is continuous
        forecast_x.insert(0, history["timestamp"].iloc[-1])
        forecast_y.insert(0, history[core.TARGET].iloc[-1])

    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, mode="lines+markers", name="Next 48 hours (prediction)",
        line=dict(width=2.5, dash="dash"), marker=dict(size=4)))
    fig.update_layout(
        title="Historical and predicted solar output - Pune",
        xaxis_title="Date and time", yaxis_title="Solar output (units, ~kW scale)",
        template="plotly_white", hovermode="x unified", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.sidebar.header("Controls")
chart_option = st.sidebar.radio("Chart range", ["Past 7 days + next 48 hours", "Next 48 hours only"])
table_hours = st.sidebar.selectbox("Hours shown in table", [24, 48])
if st.sidebar.button("Refresh weather forecast"):
    cached_live_forecast.clear()

st.title("☀️ Solar Power Prediction Application")
st.write("Hourly solar output predictions for Pune, Maharashtra, using a live weather forecast.")

# 1) model + history
try:
    bundle, history, metrics = load_model_and_history()
except Exception as error:
    st.error(f"Could not load the trained project files: {error}")
    st.info("Run `python train.py` (or all notebook cells) first. It creates "
            "`models/solar_model.joblib` and `solar_weather_dataset.csv`. "
            "Then start the app from the same folder.")
    st.stop()

# 2) weather forecast (live, or saved copy if the API is down)
try:
    with st.spinner("Getting the latest Pune weather forecast..."):
        weather, source = get_forecast(bundle["fill_values"])
except Exception as error:
    st.error(f"Could not get the weather forecast: {error}")
    st.info("Check your internet connection and press 'Refresh weather forecast'.")
    st.stop()

if source == "live":
    st.success("Live weather forecast loaded from Open-Meteo.")
else:
    st.warning(f"Weather API unavailable - showing the last saved forecast, which may be out of date. ({source})")

forecast = core.predict(bundle, weather)

# 3) summary cards
next_24 = forecast.head(24)
peak = next_24.loc[next_24["predicted_solar_output"].idxmax()]
st.subheader(f"Peak predicted output: {peak['predicted_solar_output']:.2f}")
st.caption(f"Expected peak time: {peak['timestamp']:%d %b %Y, %H:%M}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total output (next 24 h)", f"{next_24['predicted_solar_output'].sum():,.0f} unit-hours")
c2.metric(f"Productive hours (> {PRODUCTIVE_THRESHOLD})",
          f"{int((next_24['predicted_solar_output'] > PRODUCTIVE_THRESHOLD).sum())} / {len(next_24)}")
c3.metric("Average cloud cover", f"{next_24['cloud_cover'].mean():.1f}%")
c4.metric("Average temperature", f"{next_24['temperature'].mean():.1f} °C")

# 4) chart
st.subheader("Solar output chart")
show_history = chart_option.startswith("Past")
history_for_chart = shift_history_for_chart(history, forecast["timestamp"].iloc[0])
st.plotly_chart(build_chart(history_for_chart, forecast, show_history), width="stretch")
st.caption("Historical dates are shifted forward for display only (the data is from Jan-Feb 2024); "
           "the original values are unchanged.")

# 5) table + download
st.subheader("Hourly prediction table")
shown = forecast.head(table_hours)
table = pd.DataFrame({
    "Date and time": shown["timestamp"].dt.strftime("%d %b %Y, %H:%M"),
    "Predicted output": shown["predicted_solar_output"],
    "Temperature (°C)": shown["temperature"].round(1),
    "Humidity (%)": shown["humidity"].round(1),
    "Cloud cover (%)": shown["cloud_cover"].round(1),
    "Wind speed (m/s)": shown["wind_speed"].round(1),
    "Precipitation (mm)": shown["precipitation"].round(2),
})
st.dataframe(table, width="stretch", hide_index=True)

csv_data = forecast[["timestamp"] + core.WEATHER_COLUMNS + ["predicted_solar_output"]].copy()
csv_data["timestamp"] = csv_data["timestamp"].astype(str)
st.download_button("Download predictions as CSV", data=csv_data.to_csv(index=False),
                   file_name="solar_power_predictions.csv", mime="text/csv")

# 6) model evaluation
st.subheader(f"Model evaluation - {bundle['model_name']} on the last 10 days (240 rows)")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics['MAE']:.2f}")
m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
m3.metric("R² score", f"{metrics['R2']:.4f}")
st.caption("Trained on the first 50 days and evaluated on the last 10 days. The dataset is synthetic, "
           "so these scores show project-demo performance, not real solar-plant accuracy.")