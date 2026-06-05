import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions  (derived from real CSV headers across all three cities)
# ---------------------------------------------------------------------------

# All seven present in every city; only `bathrooms` has NaN in real data.
NUMERIC_FEATURES: List[str] = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "review_scores_rating",
    "amenities_count",
]

# OHE for Cosine vector includes neighbourhood; Jaccard excludes it to avoid
# zero-vectors when a query neighbourhood is unseen.
CATEGORICAL_ALL: List[str] = ["room_type", "property_type", "neighbourhood_cleansed"]
CATEGORICAL_JACCARD: List[str] = ["room_type", "property_type"]

AMENITY_FLAGS: List[str] = [
    "has_wifi", "has_kitchen", "has_tv", "has_ac",
    "has_pool", "has_parking", "has_washer", "has_elevator",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CityData:
    df: pd.DataFrame            # original rows, reset_index — used to retrieve listing details
    price: np.ndarray           # shape (n,) float64 — K-Means target and recommendation output

    cosine_matrix: np.ndarray   # (n, d_c)  scaled numerics + OHE(all cats) + amenity flags
    jaccard_matrix: np.ndarray  # (n, d_j)  OHE(room + property) + amenity flags  [binary only]
    knn_matrix: np.ndarray      # (n, d_k)  scaled numerics only

    numeric_scaler: StandardScaler
    numeric_cols: List[str]     # column names the scaler was fitted on, in order

    ohe_all: OneHotEncoder      # fitted on room_type + property_type + neighbourhood_cleansed
    ohe_jaccard: OneHotEncoder  # fitted on room_type + property_type


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NUMERIC_FEATURES columns that exist in df, warn about any absent."""
    present = [c for c in NUMERIC_FEATURES if c in df.columns]
    absent = set(NUMERIC_FEATURES) - set(present)
    if absent:
        logger.warning("Numeric columns absent from dataset, skipping: %s", sorted(absent))
    return df[present].copy()


def _fill_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with per-column median. Only `bathrooms` has real NaN in practice."""
    return df_num.fillna(df_num.median(numeric_only=True))


def _fill_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].fillna("Unknown")


def _make_ohe() -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_city(df: pd.DataFrame) -> CityData:
    """Fit all transformers on df and build the three similarity matrices.

    Called once per city at startup (or after a cache miss).
    The returned CityData is pickled to disk by the clustering module.
    """
    logger.info("Preprocessing %d rows ...", len(df))

    # --- Numeric features ---
    df_num = _fill_numeric(_select_numeric(df))
    numeric_cols = list(df_num.columns)

    scaler = StandardScaler()
    num_scaled: np.ndarray = scaler.fit_transform(df_num.values)   # (n, d_k)

    # --- Categorical encoders ---
    ohe_all = _make_ohe()
    cat_all_enc: np.ndarray = ohe_all.fit_transform(
        _fill_categorical(df, CATEGORICAL_ALL).values
    )                                                               # (n, d_ohe_all)

    ohe_jac = _make_ohe()
    cat_jac_enc: np.ndarray = ohe_jac.fit_transform(
        _fill_categorical(df, CATEGORICAL_JACCARD).values
    )                                                               # (n, d_ohe_jac)

    # --- Amenity flags ---
    flags: np.ndarray = df[AMENITY_FLAGS].fillna(0).astype(float).values   # (n, 8)

    # --- Assemble the three matrices ---
    cosine_matrix  = np.hstack([num_scaled, cat_all_enc, flags])
    jaccard_matrix = np.hstack([cat_jac_enc, flags])
    knn_matrix     = num_scaled.copy()

    # --- Price (target only — never in feature vectors) ---
    price_col = pd.to_numeric(df["price_numeric"], errors="coerce")
    price: np.ndarray = price_col.fillna(price_col.median()).values.astype(float)

    logger.info(
        "Preprocessing done — cosine_dim=%d  jaccard_dim=%d  knn_dim=%d",
        cosine_matrix.shape[1], jaccard_matrix.shape[1], knn_matrix.shape[1],
    )

    return CityData(
        df=df.reset_index(drop=True),
        price=price,
        cosine_matrix=cosine_matrix,
        jaccard_matrix=jaccard_matrix,
        knn_matrix=knn_matrix,
        numeric_scaler=scaler,
        numeric_cols=numeric_cols,
        ohe_all=ohe_all,
        ohe_jaccard=ohe_jac,
    )


def preprocess_query(features: dict, city_data: CityData) -> Dict[str, np.ndarray]:
    """Transform a single query listing into the same feature spaces as the city.

    `features` is a plain dict extracted by the Gemini call, for example:
        {
            "accommodates": 4,
            "bathrooms": 1.0,
            "bedrooms": 2,
            "beds": 2,
            "minimum_nights": 1,
            "review_scores_rating": None,
            "amenities_count": 3,
            "room_type": "Entire home/apt",
            "property_type": "Apartment",
            "neighbourhood_cleansed": "Centre",
            "has_wifi": 1, "has_kitchen": 1, "has_tv": 0,
            "has_ac": 0,   "has_pool": 0,    "has_parking": 0,
            "has_washer": 1, "has_elevator": 0,
        }

    Unknown numeric fields → city mean (z-score = 0, neutral in feature space).
    Unknown categorical fields → "Unknown" (OHE handle_unknown="ignore" → all zeros).
    Unknown amenity flags → 0.

    Returns {"cosine": (1, d_c), "jaccard": (1, d_j), "knn": (1, d_k)}.
    """
    # --- Numeric ---
    num_vals = []
    for i, col in enumerate(city_data.numeric_cols):
        val = features.get(col)
        if val is None:
            # City mean → z-score of 0; keeps the query at a neutral position
            # for that dimension rather than at an outlier extreme.
            num_vals.append(float(city_data.numeric_scaler.mean_[i]))
        else:
            num_vals.append(float(val))

    num_arr: np.ndarray = city_data.numeric_scaler.transform([num_vals])  # (1, d_k)

    # --- Categorical ---
    cat_all_arr: np.ndarray = city_data.ohe_all.transform([[
        features.get("room_type") or "Unknown",
        features.get("property_type") or "Unknown",
        features.get("neighbourhood_cleansed") or "Unknown",
    ]])                                                                    # (1, d_ohe_all)

    cat_jac_arr: np.ndarray = city_data.ohe_jaccard.transform([[
        features.get("room_type") or "Unknown",
        features.get("property_type") or "Unknown",
    ]])                                                                    # (1, d_ohe_jac)

    # --- Amenity flags ---
    flags: np.ndarray = np.array(
        [[float(features.get(f) or 0) for f in AMENITY_FLAGS]]
    )                                                                      # (1, 8)

    return {
        "cosine":  np.hstack([num_arr, cat_all_arr, flags]),
        "jaccard": np.hstack([cat_jac_arr, flags]),
        "knn":     num_arr,
    }
