"""House Price Predictor - main analysis script (Day 7).

Pipeline (runs end to end):
    1. Generate (or reuse) synthetic housing data    -> generate_data.py
    2. One-hot encode zone / property type via a ColumnTransformer pipeline
    3. Train GradientBoostingRegressor on an 80/20 split
    4. Evaluate: R2 / RMSE / MAE vs a "predict the mean" baseline
    5. Permutation importance on the held-out test set -> honest driver ranking
    6. Save charts                                  -> charts/*.png
    7. Save machine-readable summary                 -> findings.json
    8. Generate the plain-English AI narrative       -> ai_insights.py

Usage:  python3 house_price_predictor.py
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: save figures, don't open windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import generate_data
import ai_insights

DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "houses.csv")
CHART_DIR = os.path.join(os.path.dirname(__file__), "charts")
FINDINGS = os.path.join(os.path.dirname(__file__), "findings.json")

CATEGORICAL = ["zone", "property_type"]
TARGET = "price_lakhs"

sns.set_style("whitegrid")


def load_data() -> pd.DataFrame:
    """Load the CSV, generating it first if it is missing."""
    if not os.path.exists(DATA_CSV):
        print("Data not found, generating...")
        generate_data.main()
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df):,} listings, {df.shape[1]-1} features -> {TARGET}")
    return df


def build_pipeline() -> ColumnTransformer:
    """One-hot encode categoricals, pass numerics through untouched."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )


def train_model(X_train, y_train) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.06,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,            # stochastic boosting: less overfitting
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, y_train) -> dict:
    pred = model.predict(X_test)
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    return {
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "price_mean": float(y_test.mean()),
        "y_test": y_test.to_numpy(),
        "y_pred": pred,
    }


def rank_drivers(model, X_test, y_test, feature_names) -> list:
    """Permutation importance: drop in R2 when each feature is scrambled."""
    imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="r2", n_jobs=-1,
    )
    order = np.argsort(imp.importances_mean)[::-1]
    return [
        {"feature": feature_names[i],
         "drop_in_r2": float(imp.importances_mean[i])}
        for i in order
    ]


def save_charts(df, y_test, y_pred, drivers) -> None:
    os.makedirs(CHART_DIR, exist_ok=True)

    # 1. Actual vs predicted
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_test, y_pred, alpha=0.35, s=14)
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual price (INR lakhs)")
    ax.set_ylabel("Predicted price (INR lakhs)")
    ax.set_title("Actual vs Predicted House Prices (test set)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{CHART_DIR}/actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # 2. Residuals
    resid = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, resid, alpha=0.35, s=14)
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("Predicted price (INR lakhs)")
    ax.set_ylabel("Residual = actual - predicted (INR lakhs)")
    ax.set_title("Residual Plot (should scatter evenly around 0)")
    fig.tight_layout(); fig.savefig(f"{CHART_DIR}/residual_plot.png", dpi=120)
    plt.close(fig)

    # 3. Permutation importance
    top = drivers[:8][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([d["feature"] for d in top], [d["drop_in_r2"] for d in top])
    ax.set_xlabel("Mean drop in R² when feature is scrambled")
    ax.set_title("What drives the price? (permutation importance)")
    fig.tight_layout(); fig.savefig(f"{CHART_DIR}/permutation_importance.png", dpi=120)
    plt.close(fig)

    # 4. Price distribution by zone
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="zone", y="price_lakhs",
                order=["Central", "North", "East", "West", "South"], ax=ax)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Price (INR lakhs)")
    ax.set_title("Price distribution by zone")
    fig.tight_layout(); fig.savefig(f"{CHART_DIR}/price_by_zone.png", dpi=120)
    plt.close(fig)
    print(f"Saved 4 charts -> {CHART_DIR}/")


def main() -> None:
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    pre = build_pipeline()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    feature_names = list(pre.get_feature_names_out())
    print(f"Encoded to {len(feature_names)} features (one-hot + numeric)")

    model = train_model(X_train_t, y_train)
    metrics = evaluate(model, X_test_t, y_test, y_train)
    print(f"Test R² = {metrics['test_r2']:.4f} | RMSE = INR {metrics['test_rmse']:.2f}L "
          f"| MAE = INR {metrics['test_mae']:.2f}L "
          f"(baseline RMSE = INR {metrics['baseline_rmse']:.2f}L)")

    drivers = rank_drivers(model, X_test_t, y_test, feature_names)
    print("Top 5 drivers:", ", ".join(d["feature"] for d in drivers[:5]))

    save_charts(df, metrics.pop("y_test"), metrics.pop("y_pred"), drivers)

    findings = {
        "n_rows": len(df),
        "test_r2": metrics["test_r2"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "baseline_rmse": metrics["baseline_rmse"],
        "price_mean": metrics["price_mean"],
        "permutation_importance": drivers,
    }
    with open(FINDINGS, "w") as fh:
        json.dump(findings, fh, indent=2)
    print(f"Saved metrics -> {FINDINGS}")

    ai_insights.write_insights(findings)
    print("Done.")


if __name__ == "__main__":
    main()
