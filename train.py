"""
train.py - train, tune, evaluate and save the solar power model.

Usage:  python train.py
Creates: solar_weather_dataset.csv (if missing), models/solar_model.joblib, outputs/model_metrics.json
"""
import json

import joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

import solar_core as core


def main():
    # 1. data -> clean -> features
    if not core.DATA_FILE.exists():
        core.generate_dataset(core.DATA_FILE)
        print("Dataset created:", core.DATA_FILE.name)
    df = core.add_time_features(core.load_and_clean(core.DATA_FILE))

    # 2. chronological split: first 50 days train, last 10 days validation
    train, valid = df.iloc[:core.TRAIN_ROWS], df.iloc[core.TRAIN_ROWS:]
    X_train, y_train = train[core.FEATURES], train[core.TARGET]
    X_valid, y_valid = valid[core.FEATURES], valid[core.TARGET]
    print(f"Train: {len(train)} rows | Validation: {len(valid)} rows")

    # 3. tune three models (time-ordered cross-validation inside the training days only)
    models = {
        "Random Forest": (RandomForestRegressor(random_state=42),
                          {"n_estimators": [100, 200], "max_depth": [8, 15]}),
        "XGBoost": (XGBRegressor(random_state=42, verbosity=0),
                    {"n_estimators": [100, 200], "max_depth": [3, 5]}),
        "LightGBM": (LGBMRegressor(random_state=42, verbose=-1),
                     {"n_estimators": [100, 200], "num_leaves": [15, 31]}),
    }
    tuned = {}
    for name, (model, grid) in models.items():
        search = GridSearchCV(model, grid, cv=TimeSeriesSplit(n_splits=3),
                              scoring="neg_root_mean_squared_error")
        search.fit(X_train, y_train)
        tuned[name] = search
        print(f"{name:14s} CV RMSE = {-search.best_score_:.2f}  params = {search.best_params_}")

    # 4. select by cross-validation RMSE, then score on the validation days
    best_name = min(tuned, key=lambda n: -tuned[n].best_score_)
    best_model = tuned[best_name].best_estimator_
    metrics = core.evaluate(best_model, X_valid, y_valid)
    print(f"\nSelected: {best_name}")
    print(f"Validation  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.4f}")

    # 5. save model bundle + metrics
    core.MODEL_FILE.parent.mkdir(exist_ok=True)
    core.METRICS_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name, "features": core.FEATURES,
                 "fill_values": X_train.median().to_dict()}, core.MODEL_FILE)
    core.METRICS_FILE.write_text(json.dumps(
        {"model": best_name, **{k: round(v, 4) for k, v in metrics.items()}}, indent=2))
    print("Saved:", core.MODEL_FILE)


if __name__ == "__main__":
    main()