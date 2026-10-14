"""Synthetic housing dataset generator (Day 7 - fixed seed 42).

Builds a reproducible CSV of house listings with a *known* ground-truth
pricing formula. Because the formula is known, the trained model can be
sanity-checked: a good model should rediscover the same drivers.

Columns:
    area_sqft, bedrooms, bathrooms, age_years, distance_km, floor,
    has_parking, has_balcony_garden, zone, property_type -> price_lakhs
"""

import os

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2500   # keep data/houses.csv (~110 KB) pushable in one commit
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "houses.csv")

ZONES = ["Central", "North", "East", "West", "South"]
ZONE_W = [0.18, 0.18, 0.22, 0.20, 0.22]
PTYPES = ["apartment", "independent_house", "villa", "studio"]
PTYPE_W = [0.55, 0.20, 0.10, 0.15]

# Ground-truth pricing components (lakhs of INR).
ZONE_PREMIUM = {"Central": 22.0, "North": 8.0, "East": 4.0, "West": 6.0, "South": 5.0}
PTYPE_PREMIUM = {"apartment": 0.0, "independent_house": 10.0, "villa": 45.0, "studio": -6.0}


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    zone = rng.choice(ZONES, n, p=ZONE_W)
    ptype = rng.choice(PTYPES, n, p=PTYPE_W)

    # Size depends on property type (studios small, villas big).
    area_lo = {"apartment": 600, "independent_house": 900, "villa": 1800, "studio": 400}
    area_hi = {"apartment": 2200, "independent_house": 3000, "villa": 3500, "studio": 900}
    area = np.array([rng.integers(area_lo[t], area_hi[t] + 1) for t in ptype], dtype=float)

    bedrooms = np.clip(np.round(area / 500 + rng.normal(0, 0.4, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5).astype(int)
    age = rng.integers(0, 41, n).astype(float)
    distance = np.clip(rng.gamma(2.0, 3.5, n), 0.5, 25.0).round(1)   # km to city centre
    floor = np.where(np.isin(ptype, ["independent_house", "villa"]),
                     0, rng.integers(0, 21, n)).astype(int)
    parking = rng.binomial(1, 0.65, n).astype(int)
    garden = rng.binomial(1, 0.50, n).astype(int)

    # --- Ground-truth price (the model must rediscover this) -----------------
    # Per-sqft rate with diminishing returns for very large homes; a small
    # premium for higher floors; everything else linear.
    price = (
        6.0
        + 0.048 * area
        - 0.000012 * area ** 2            # diminishing returns on size
        + 5.5 * bedrooms
        + 3.5 * bathrooms
        - 0.32 * age
        - 2.0 * distance
        + 4.5 * parking
        + 2.5 * garden
        + 0.6 * np.sqrt(floor + 1)
        + np.array([ZONE_PREMIUM[z] for z in zone])
        + np.array([PTYPE_PREMIUM[t] for t in ptype])
        + rng.normal(0, 3.5, n)           # market noise
    )
    price = np.maximum(price, 10.0).round(1)

    return pd.DataFrame({
        "area_sqft": area.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age.astype(int),
        "distance_km": distance,
        "floor": floor,
        "has_parking": parking,
        "has_balcony_garden": garden,
        "zone": zone,
        "property_type": ptype,
        "price_lakhs": price,
    })


def main() -> None:
    df = generate()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df):,} rows -> {OUTPUT_CSV}")
    print(f"Price range: INR {df.price_lakhs.min():.1f}L - {df.price_lakhs.max():.1f}L "
          f"(mean {df.price_lakhs.mean():.1f}L)")


if __name__ == "__main__":
    main()
