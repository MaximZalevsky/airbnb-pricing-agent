import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from config.settings import CROSS_CITY_CONFIDENCE_PENALTY, RECOMMENDATION_COMPS
from engine.recommender import Recommendation, recommend

logger = logging.getLogger(__name__)


@dataclass
class CrossCityRecommendation:
    rec: Recommendation
    source_cities: List[str]


def _price_consistency(comp_prices: np.ndarray) -> float:
    if len(comp_prices) < 2 or comp_prices.mean() == 0:
        return 0.5
    cv = float(comp_prices.std() / comp_prices.mean())
    return float(max(0.0, 1.0 - cv))


def recommend_cross_city(
    query_features: dict,
    requested_city: str,
    cities: dict,
    n_comps: int = RECOMMENDATION_COMPS,
) -> CrossCityRecommendation:
    """Cross-city fallback when the requested city has no dataset.

    Runs recommend() for every loaded city, pools comparables tagged with
    source_city, returns the top-N by similarity with penalized confidence.
    """
    all_comparables: list = []

    for city_name, city_state in cities.items():
        try:
            sub_rec = recommend(
                query_features=query_features,
                city_name=city_name,
                city_data=city_state["data"],
                index=city_state["index"],
            )
            for comp in sub_rec.comparables:
                all_comparables.append({**comp, "source_city": city_name})
        except Exception:
            logger.warning("Cross-city: recommend() failed for '%s', skipping.", city_name)

    all_comparables.sort(key=lambda c: c["similarity_score"], reverse=True)
    top_comps = all_comparables[:n_comps]
    source_cities = sorted({c["source_city"] for c in top_comps})

    comp_prices = np.array([c["price"] for c in top_comps], dtype=float)

    if len(comp_prices) == 0:
        recommended_price = price_min = price_max = 0.0
        confidence = 0.0
    else:
        recommended_price = float(np.median(comp_prices))
        price_min = float(comp_prices.min())
        price_max = float(comp_prices.max())
        consistency = _price_consistency(comp_prices)
        coverage = len(comp_prices) / max(n_comps, 1)
        raw_confidence = 0.60 * consistency + 0.40 * coverage
        confidence = round(float(raw_confidence * CROSS_CITY_CONFIDENCE_PENALTY), 3)

    confidence_pct = int(round(confidence * 100))

    rec = Recommendation(
        city_name=requested_city,
        recommended_price=recommended_price,
        price_min=price_min,
        price_max=price_max,
        price_tier=None,
        tier_label="unknown",
        confidence=confidence,
        confidence_pct=confidence_pct,
        comparables=top_comps,
    )

    logger.info(
        "Cross-city for '%s': %.0f | sources=%s | confidence=%d%%",
        requested_city, recommended_price, source_cities, confidence_pct,
    )

    return CrossCityRecommendation(rec=rec, source_cities=source_cities)
