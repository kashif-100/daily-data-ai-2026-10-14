# Day 7 — House Price Predictor

Predict house prices from listing features using **Gradient Boosting
regression**, then explain *why* each prediction costs what it does with
**permutation feature importance**. A template-driven narrative generator
(`ai_insights.py`) turns the model metrics into a plain-English AI valuation
summary — that plus the ML model is the "AI integrated" part.

## What it does

1. `generate_data.py` — builds 2,500 synthetic house listings from a
   **known** ground-truth pricing formula (fixed seed `42`, reproducible).
   Features: area, bedrooms, bathrooms, age, distance to city centre, floor,
   parking, balcony/garden, zone, property type. Target: `price_lakhs`.
   No external APIs.
2. `house_price_predictor.py` — the main pipeline:
   - one-hot encodes zone / property type with a `ColumnTransformer`
   - trains `GradientBoostingRegressor` on an 80/20 split
   - evaluates R² / RMSE / MAE against a "predict the mean" baseline
   - ranks price drivers with permutation importance on the **held-out test set**
   - saves 4 charts to `charts/` and a machine-readable `findings.json`
   - calls `ai_insights.py` to write `AI_INSIGHTS.md`
3. `ai_insights.py` — generates the plain-English valuation narrative from
   `findings.json`. Deterministic and template-based: it can only state
   numbers the model actually measured, so it cannot hallucinate figures.

## Key findings

- **2,500 listings, INR 10.0L–172.2L (mean INR 71.1L).**
- **Test R² = 0.983, RMSE = INR 4.0L, MAE = INR 3.2L** — typical prediction
  error is ~5.6% of the average price. The "guess the average" baseline
  misses by INR 30.7L, so the model cuts error by ~87%.
- **Price drivers (permutation importance):** distance to city centre (0.212 R² drop) >
  area (0.197) > being a villa (0.147) > bedrooms (0.141) > Central zone
  (0.079). Location and size dominate — matching the ground-truth formula,
  which is a strong sanity check that the model learned real structure.
- **Residuals scatter evenly around zero** with no obvious pattern, meaning
  the model is not systematically over- or under-predicting any price band.

Full narrative with buyer/seller takeaways: see `AI_INSIGHTS.md`.

## How to run

```bash
pip install -r requirements.txt
python3 house_price_predictor.py   # regenerates data if data/houses.csv is missing
```

Outputs: `charts/*.png`, `findings.json`, `AI_INSIGHTS.md`.
