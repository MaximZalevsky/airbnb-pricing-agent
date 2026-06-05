"""
Tests for Phase 4 — cross-city fallback recommendation.

All tests are offline: they mock recommend() and _call_gemini so no real
dataset or Gemini API key is needed.

Run:
    py -m pytest tests/test_cross_city.py -v
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stub for Recommendation (avoids importing ML deps in unit tests)
# ---------------------------------------------------------------------------

@dataclass
class _FakeRec:
    city_name: str
    recommended_price: float
    price_min: float
    price_max: float
    price_tier: Optional[int]
    tier_label: str
    confidence: float
    confidence_pct: int
    comparables: List[dict] = field(default_factory=list)


def _make_comp(name: str, price: float, score: float) -> dict:
    return {
        "name": name,
        "neighbourhood": "TestArea",
        "room_type": "Entire home/apt",
        "property_type": "Apartment",
        "accommodates": 2,
        "price": price,
        "price_tier": 1,
        "similarity_score": score,
    }


def _make_fake_rec(city_name: str) -> _FakeRec:
    """Direct helper for tests that build a stub rec without going through mock dispatch."""
    base = {"amsterdam": 100.0, "london": 150.0, "rome": 80.0}.get(city_name, 120.0)
    comps = [
        _make_comp(f"{city_name}-A", base,       0.90),
        _make_comp(f"{city_name}-B", base + 20,  0.80),
        _make_comp(f"{city_name}-C", base - 10,  0.70),
    ]
    return _FakeRec(
        city_name=city_name,
        recommended_price=base,
        price_min=base - 10,
        price_max=base + 20,
        price_tier=1,
        tier_label="mid-range",
        confidence=0.75,
        confidence_pct=75,
        comparables=comps,
    )


def _fake_rec_for_city(query_features, city_name, city_data, index, expected_price=None) -> _FakeRec:
    """Mock side_effect with the full recommend() signature."""
    base = {"amsterdam": 100.0, "london": 150.0, "rome": 80.0}.get(city_name, 120.0)
    comps = [
        _make_comp(f"{city_name}-A", base,       0.90),
        _make_comp(f"{city_name}-B", base + 20,  0.80),
        _make_comp(f"{city_name}-C", base - 10,  0.70),
    ]
    return _FakeRec(
        city_name=city_name,
        recommended_price=base,
        price_min=base - 10,
        price_max=base + 20,
        price_tier=1,
        tier_label="mid-range",
        confidence=0.75,
        confidence_pct=75,
        comparables=comps,
    )


_CITIES = {
    "amsterdam": {"data": MagicMock(), "index": MagicMock()},
    "london":    {"data": MagicMock(), "index": MagicMock()},
    "rome":      {"data": MagicMock(), "index": MagicMock()},
}

_QUERY_FEATURES = {
    "room_type": "Entire home/apt",
    "accommodates": 2,
}


# ---------------------------------------------------------------------------
# Tests for engine/cross_city.py
# ---------------------------------------------------------------------------

class TestRecommendCrossCity:

    def _run(self, n_comps=5):
        from engine.cross_city import recommend_cross_city
        with patch("engine.cross_city.recommend", side_effect=_fake_rec_for_city):
            return recommend_cross_city(_QUERY_FEATURES, "barcelona", _CITIES, n_comps=n_comps)

    def test_returns_cross_city_recommendation(self):
        from engine.cross_city import CrossCityRecommendation
        result = self._run()
        assert isinstance(result, CrossCityRecommendation)

    def test_inner_rec_is_recommendation_type(self):
        from engine.recommender import Recommendation
        result = self._run()
        assert isinstance(result.rec, Recommendation)

    def test_requested_city_name_preserved(self):
        result = self._run()
        assert result.rec.city_name == "barcelona"

    def test_confidence_at_most_55_pct(self):
        result = self._run()
        assert result.rec.confidence_pct <= 55, (
            f"Expected ≤55%, got {result.rec.confidence_pct}%"
        )

    def test_confidence_raw_value_at_most_0_55(self):
        result = self._run()
        assert result.rec.confidence <= 0.55

    def test_all_comparables_have_source_city(self):
        result = self._run()
        for comp in result.rec.comparables:
            assert "source_city" in comp, f"Comparable missing source_city: {comp}"

    def test_source_city_values_are_valid_cities(self):
        result = self._run()
        for comp in result.rec.comparables:
            assert comp["source_city"] in _CITIES

    def test_source_cities_list_populated(self):
        result = self._run()
        assert len(result.source_cities) > 0

    def test_source_cities_are_subset_of_loaded_cities(self):
        result = self._run()
        assert set(result.source_cities).issubset(set(_CITIES.keys()))

    def test_recommend_called_once_per_city(self):
        from engine.cross_city import recommend_cross_city
        with patch("engine.cross_city.recommend", side_effect=_fake_rec_for_city) as mock_rec:
            recommend_cross_city(_QUERY_FEATURES, "barcelona", _CITIES, n_comps=5)
        assert mock_rec.call_count == len(_CITIES)

    def test_comparables_sorted_by_score_descending(self):
        result = self._run()
        scores = [c["similarity_score"] for c in result.rec.comparables]
        assert scores == sorted(scores, reverse=True)

    def test_recommended_price_is_median_of_top_comps(self):
        import numpy as np
        result = self._run()
        prices = [c["price"] for c in result.rec.comparables]
        assert abs(result.rec.recommended_price - float(np.median(prices))) < 0.01

    def test_n_comps_respected(self):
        result = self._run(n_comps=4)
        assert len(result.rec.comparables) <= 4

    def test_failed_city_skipped_gracefully(self):
        from engine.cross_city import recommend_cross_city

        def _side_effect(query_features, city_name, city_data, index, expected_price=None):
            if city_name == "rome":
                raise RuntimeError("simulated failure")
            return _fake_rec_for_city(query_features, city_name, city_data, index)

        with patch("engine.cross_city.recommend", side_effect=_side_effect):
            result = recommend_cross_city(_QUERY_FEATURES, "barcelona", _CITIES)

        # Should still return a result without crashing
        assert result.rec.recommended_price > 0
        for comp in result.rec.comparables:
            assert comp["source_city"] != "rome"

    def test_tier_label_is_unknown(self):
        result = self._run()
        assert result.rec.tier_label == "unknown"

    def test_price_tier_is_none(self):
        result = self._run()
        assert result.rec.price_tier is None


# ---------------------------------------------------------------------------
# Tests for nlp/gemini.py — disclaimer injection
# ---------------------------------------------------------------------------

class TestDisclaimerInjection:

    def _run_explanation(self, lang: str, is_cross_city: bool, source_cities=None):
        import nlp.gemini as gem

        rec = _make_fake_rec("amsterdam")
        captured_system = {}

        def _fake_call_gemini(client, system, user_prompt, json_mode=False):
            captured_system["system"] = system
            return "Mocked explanation text."

        with patch("nlp.gemini._call_gemini", side_effect=_fake_call_gemini):
            gem.generate_explanation(
                client=MagicMock(),
                city="barcelona",
                features=_QUERY_FEATURES,
                rec=rec,
                user_language=lang,
                is_cross_city=is_cross_city,
                source_cities=source_cities or ["amsterdam", "london"],
            )
        return captured_system.get("system", "")

    def test_no_disclaimer_for_in_dataset_city(self):
        system = self._run_explanation("he", is_cross_city=False)
        assert "אין לנו" not in system
        assert "do not currently have" not in system.lower()

    def test_hebrew_disclaimer_present_when_cross_city(self):
        system = self._run_explanation("he", is_cross_city=True, source_cities=["amsterdam"])
        assert "אין לנו" in system

    def test_english_disclaimer_present_when_cross_city(self):
        system = self._run_explanation("en", is_cross_city=True, source_cities=["amsterdam"])
        assert "do not currently have" in system.lower()

    def test_source_cities_appear_in_disclaimer(self):
        system = self._run_explanation("en", is_cross_city=True, source_cities=["amsterdam", "london"])
        assert "Amsterdam" in system or "amsterdam" in system.lower()

    def test_requested_city_in_disclaimer(self):
        system = self._run_explanation("he", is_cross_city=True)
        assert "Barcelona" in system or "barcelona" in system.lower()

    def test_signature_backwards_compatible(self):
        """generate_explanation must be callable without the new kwargs (existing callers)."""
        import nlp.gemini as gem
        rec = _make_fake_rec("amsterdam")
        with patch("nlp.gemini._call_gemini", return_value="ok"):
            result = gem.generate_explanation(MagicMock(), "amsterdam", _QUERY_FEATURES, rec)
        assert result == "ok"
