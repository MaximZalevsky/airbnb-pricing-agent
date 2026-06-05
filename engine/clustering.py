import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans

from config.settings import (
    CACHE_DIR,
    KMEANS_K_LARGE,
    KMEANS_K_MEDIUM,
    KMEANS_K_SMALL,
    KMEANS_RANDOM_STATE,
    KMEANS_THRESH_MEDIUM,
    KMEANS_THRESH_SMALL,
)
from data.preprocessor import CityData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adaptive_k(n_rows: int) -> int:
    if n_rows < KMEANS_THRESH_SMALL:
        return KMEANS_K_SMALL
    if n_rows < KMEANS_THRESH_MEDIUM:
        return KMEANS_K_MEDIUM
    return KMEANS_K_LARGE


def _cache_path(city_name: str) -> Path:
    return CACHE_DIR / f"{city_name}_kmeans.pkl"


def _save_pkl(obj: object, path: Path) -> None:
    """Write to a .tmp file then atomically replace the target.

    Prevents a corrupt .pkl if two Passenger workers race on first startup.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_pkl(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def _fit(price: np.ndarray, k: int) -> Tuple[KMeans, Dict[int, int], np.ndarray]:
    """Fit KMeans on price (1-D). Returns (model, remap, tier_labels).

    remap translates raw cluster IDs to tier numbers sorted cheapest→priciest,
    so tier 0 is always the budget segment regardless of KMeans label assignment.
    """
    kmeans = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init="auto")
    raw_labels: np.ndarray = kmeans.fit_predict(price.reshape(-1, 1))

    # Sort cluster centroids ascending; the i-th cheapest centroid becomes tier i.
    order = np.argsort(kmeans.cluster_centers_.flatten())
    remap: Dict[int, int] = {int(old): int(new) for new, old in enumerate(order)}

    tier_labels = np.array([remap[int(lbl)] for lbl in raw_labels], dtype=np.int8)
    return kmeans, remap, tier_labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_city(city_data: CityData, city_name: str) -> CityData:
    """Assign a price tier to every listing in city_data.df.

    Tiers are integers 0 … K-1 where 0 = cheapest segment.
    K is chosen adaptively from the dataset size (settings.KMEANS_K_*).

    The fitted KMeans and remap are cached to CACHE_DIR/{city_name}_kmeans.pkl
    so subsequent restarts skip re-fitting.  Cache is invalidated automatically
    if the adaptive K changes (e.g. after a threshold setting change).

    Mutates city_data.df in place (adds 'price_tier' column) and returns it.
    """
    n = len(city_data.df)
    k = _adaptive_k(n)
    cache = _cache_path(city_name)

    tier_labels: np.ndarray

    if cache.exists():
        cached = _load_pkl(cache)
        if cached.get("k") == k:
            logger.info("Cache hit — loading KMeans for '%s' (k=%d)", city_name, k)
            tier_labels = cached["tier_labels"]
        else:
            logger.info(
                "Cache K mismatch for '%s' (cached=%d, expected=%d) — refitting.",
                city_name, cached.get("k"), k,
            )
            kmeans, remap, tier_labels = _fit(city_data.price, k)
            _save_pkl({"k": k, "kmeans": kmeans, "remap": remap, "tier_labels": tier_labels}, cache)
            logger.info("KMeans re-cached → %s", cache.name)
    else:
        logger.info("Fitting KMeans (k=%d) on %d prices for '%s' ...", k, n, city_name)
        kmeans, remap, tier_labels = _fit(city_data.price, k)
        _save_pkl({"k": k, "kmeans": kmeans, "remap": remap, "tier_labels": tier_labels}, cache)
        logger.info("KMeans cached → %s", cache.name)

    city_data.df["price_tier"] = tier_labels
    return city_data


def predict_tier(price: float, city_name: str) -> int:
    """Return the price tier for a single price value at query time.

    Uses the cached KMeans fitted by cluster_city() — must be called after
    cluster_city() has run at least once for this city.
    """
    cache = _cache_path(city_name)
    if not cache.exists():
        raise FileNotFoundError(
            f"No KMeans cache found for '{city_name}'. "
            "cluster_city() must run before predict_tier()."
        )

    cached = _load_pkl(cache)
    kmeans: KMeans = cached["kmeans"]
    remap: Dict[int, int] = cached["remap"]

    raw: int = int(kmeans.predict([[price]])[0])
    return remap[raw]


def tier_price_ranges_from_data(
    city_data: CityData, city_name: str
) -> Dict[int, Tuple[float, float]]:
    """Return {tier: (min_price, max_price)} computed from live city_data.

    Requires 'price_tier' column to already exist in city_data.df
    (i.e. cluster_city() has been called).
    """
    if "price_tier" not in city_data.df.columns:
        raise ValueError(
            "'price_tier' column missing — call cluster_city() first."
        )

    ranges: Dict[int, Tuple[float, float]] = {}
    for tier in sorted(city_data.df["price_tier"].unique()):
        mask = city_data.df["price_tier"] == tier
        prices = city_data.price[mask]
        ranges[int(tier)] = (float(prices.min()), float(prices.max()))

    return ranges
