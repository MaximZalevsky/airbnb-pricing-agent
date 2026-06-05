import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import RECOMMENDATION_COMPS, SIMILARITY_TOP_K
from data.preprocessor import CityData, preprocess_query
from engine.clustering import predict_tier
from engine.similarity import SimilarityIndex, find_similar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier label helpers
# ---------------------------------------------------------------------------

_LABELS_K2 = {0: "budget", 1: "premium"}
_LABELS_K3 = {0: "budget", 1: "mid-range", 2: "premium"}
_LABELS_K4 = {0: "budget", 1: "mid-range", 2: "upscale", 3: "luxury"}

_TIER_LABEL_MAP = {2: _LABELS_K2, 3: _LABELS_K3, 4: _LABELS_K4}


def _tier_label(tier: Optional[int], k: int) -> str:
    if tier is None:
        return "unknown"
    return _TIER_LABEL_MAP.get(k, _LABELS_K4).get(tier, f"tier-{tier}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    city_name: str
    recommended_price: float          # median price of the top comparables
    price_min: float                  # lowest comparable price
    price_max: float                  # highest comparable price
    price_tier: Optional[int]         # 0-based tier index (None if no expected_price)
    tier_label: str                   # "budget", "mid-range", "upscale", "luxury"
    confidence: float                 # 0.0–1.0
    confidence_pct: int               # 0–100 (for display)
    comparables: List[dict] = field(default_factory=list)
    # Each comparable dict:
    #   name, neighbourhood, room_type, property_type,
    #   accommodates, price, price_tier, similarity_score


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    hits: Dict[str, List[Tuple[int, float]]],
    top_k: int,
) -> List[Tuple[int, float]]:
    """Rank-based score fusion across the three metrics.

    Position 0 earns top_k points, position 1 earns top_k-1, etc.
    A listing that appears in multiple metrics accumulates scores from each.
    """
    scores: Dict[int, float] = {}
    for hit_list in hits.values():
        for rank, (idx, _dist) in enumerate(hit_list):
            scores[idx] = scores.get(idx, 0.0) + float(top_k - rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _metric_agreement(
    hits: Dict[str, List[Tuple[int, float]]],
    top_indices: List[int],
) -> float:
    """Average fraction of metrics each top comparable appeared in (0–1).

    A comparable present in all 3 metrics → 1.0 contribution.
    Present in 1 of 3 → 0.33 contribution.
    """
    n_metrics = len(hits)
    if not top_indices or n_metrics == 0:
        return 0.0

    index_sets = [frozenset(idx for idx, _ in hit_list) for hit_list in hits.values()]
    counts = [
        sum(1 for s in index_sets if idx in s)
        for idx in top_indices
    ]
    return sum(counts) / (len(top_indices) * n_metrics)


def _price_consistency(comp_prices: np.ndarray) -> float:
    """Price consistency among comparables, expressed as 1 − coefficient_of_variation.

    Low spread (CV→0) → score→1.0  (high confidence: comps agree on price)
    High spread (CV≥1) → score→0.0 (low confidence: comps wildly disagree)
    """
    if len(comp_prices) < 2 or comp_prices.mean() == 0:
        return 0.5
    cv = float(comp_prices.std() / comp_prices.mean())
    return float(max(0.0, 1.0 - cv))


def _compute_confidence(
    hits: Dict[str, List[Tuple[int, float]]],
    top_indices: List[int],
    comp_prices: np.ndarray,
    n_requested: int,
) -> float:
    """Composite confidence score in [0.0, 1.0].

    Three weighted factors:
      35% — cross-metric agreement   (did multiple metrics agree on these comps?)
      45% — price consistency        (do the comparable prices cluster tightly?)
      20% — coverage                 (did we find the full requested number of comps?)
    """
    agreement   = _metric_agreement(hits, top_indices)
    consistency = _price_consistency(comp_prices)
    coverage    = len(comp_prices) / max(n_requested, 1)

    score = 0.35 * agreement + 0.45 * consistency + 0.20 * coverage
    return round(float(min(max(score, 0.0), 1.0)), 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    query_features: dict,
    city_name: str,
    city_data: CityData,
    index: SimilarityIndex,
    expected_price: Optional[float] = None,
) -> Recommendation:
    """Run the full ML recommendation pipeline for one user query.

    Parameters
    ----------
    query_features : dict extracted by the Gemini feature-extraction call.
    city_name      : lowercase city key, e.g. "amsterdam".
    city_data      : pre-processed CityData (preprocessor + clustering already run).
    index          : fitted SimilarityIndex for this city.
    expected_price : the price the user expects to charge (optional, for tier placement).

    Returns
    -------
    Recommendation — passed to the Flask route for Gemini explanation.
    All numeric fields are native Python float/int (JSON-safe).
    """
    # 1. Transform query into feature vectors
    query_vectors = preprocess_query(query_features, city_data)

    # 2. Find top-K candidates per metric
    hits = find_similar(
        query_vectors=query_vectors,
        index=index,
        n_rows=len(city_data.df),
        top_k=SIMILARITY_TOP_K,
    )

    # 3. Fuse rankings
    ranked    = _aggregate(hits, top_k=SIMILARITY_TOP_K)
    top_items = ranked[:RECOMMENDATION_COMPS]
    top_indices = [idx for idx, _ in top_items]

    # 4. Comparable prices
    comp_prices = city_data.price[top_indices]

    # 5. Price recommendation (median is robust to small-comp-set outliers)
    recommended_price = float(np.median(comp_prices))
    price_min         = float(comp_prices.min())
    price_max         = float(comp_prices.max())

    # 6. Confidence
    confidence     = _compute_confidence(hits, top_indices, comp_prices, RECOMMENDATION_COMPS)
    confidence_pct = int(round(confidence * 100))

    # 7. Tier placement
    price_tier: Optional[int] = None
    if expected_price is not None:
        try:
            price_tier = predict_tier(float(expected_price), city_name)
        except FileNotFoundError:
            logger.warning("No KMeans cache for '%s' — skipping tier placement.", city_name)

    k_tiers = (
        int(city_data.df["price_tier"].nunique())
        if "price_tier" in city_data.df.columns
        else 4
    )

    # 8. Build comparables list (all native Python types for JSON safety)
    comparables: List[dict] = []
    for df_idx, score in top_items:
        row = city_data.df.iloc[df_idx]
        comparables.append({
            "name":             str(row.get("name", "") or "Unnamed listing"),
            "neighbourhood":    str(row.get("neighbourhood_cleansed", "")),
            "room_type":        str(row.get("room_type", "")),
            "property_type":    str(row.get("property_type", "")),
            "accommodates":     int(row.get("accommodates", 0)),
            "price":            float(city_data.price[df_idx]),
            "price_tier":       (
                int(row["price_tier"]) if "price_tier" in city_data.df.columns else None
            ),
            "similarity_score": round(float(score), 2),
        })

    logger.info(
        "Recommendation for '%s': €%.0f (€%.0f–€%.0f) | tier=%s | confidence=%d%%",
        city_name, recommended_price, price_min, price_max,
        _tier_label(price_tier, k_tiers), confidence_pct,
    )

    return Recommendation(
        city_name=city_name,
        recommended_price=recommended_price,
        price_min=price_min,
        price_max=price_max,
        price_tier=price_tier,
        tier_label=_tier_label(price_tier, k_tiers),
        confidence=confidence,
        confidence_pct=confidence_pct,
        comparables=comparables,
    )
