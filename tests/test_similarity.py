"""
Tests for engine/similarity.py and the hybrid scoring in engine/recommender.py.

Run from the project root:
    pytest tests/test_similarity.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.preprocessor import CityData
from engine.recommender import _aggregate, _compute_confidence, _metric_agreement, _price_consistency
from engine.similarity import SimilarityIndex, build_index, find_similar

# ---------------------------------------------------------------------------
# Constants for synthetic data
# ---------------------------------------------------------------------------

N_ROWS       = 25
N_COSINE     = 12   # cosine_matrix width
N_JACCARD    = 6    # jaccard_matrix width (binary)
N_KNN        = 5    # knn_matrix / numeric feature width


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def city_data() -> CityData:
    """Minimal synthetic CityData with realistic shapes."""
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "name":                   [f"Listing {i}" for i in range(N_ROWS)],
        "neighbourhood_cleansed": rng.choice(["Centre", "North", "South"], N_ROWS),
        "room_type":              rng.choice(["Entire home/apt", "Private room"], N_ROWS),
        "property_type":          rng.choice(["Apartment", "House"], N_ROWS),
        "accommodates":           rng.integers(1, 8, N_ROWS).tolist(),
        "price_tier":             rng.integers(0, 4, N_ROWS).tolist(),
    })

    price          = rng.uniform(50.0, 300.0, N_ROWS)
    cosine_matrix  = rng.random((N_ROWS, N_COSINE))
    jaccard_matrix = rng.integers(0, 2, (N_ROWS, N_JACCARD)).astype(float)
    knn_matrix     = rng.standard_normal((N_ROWS, N_KNN))

    scaler = StandardScaler().fit(knn_matrix)

    ohe_all = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_all.fit(df[["room_type", "property_type", "neighbourhood_cleansed"]].values)

    ohe_jac = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_jac.fit(df[["room_type", "property_type"]].values)

    return CityData(
        df=df,
        price=price,
        cosine_matrix=cosine_matrix,
        jaccard_matrix=jaccard_matrix,
        knn_matrix=knn_matrix,
        numeric_scaler=scaler,
        numeric_cols=[f"feat_{i}" for i in range(N_KNN)],
        ohe_all=ohe_all,
        ohe_jaccard=ohe_jac,
    )


@pytest.fixture(scope="module")
def index(city_data: CityData) -> SimilarityIndex:
    return build_index(city_data)


@pytest.fixture
def first_row_query(city_data: CityData) -> dict:
    """Query vectors identical to the first training row."""
    return {
        "cosine":  city_data.cosine_matrix[0:1].copy(),
        "jaccard": city_data.jaccard_matrix[0:1].copy(),
        "knn":     city_data.knn_matrix[0:1].copy(),
    }


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_returns_similarity_index(self, city_data):
        idx = build_index(city_data)
        assert isinstance(idx, SimilarityIndex)

    def test_all_three_models_present(self, index):
        assert index.cosine_nn  is not None
        assert index.jaccard_nn is not None
        assert index.knn_nn     is not None


# ---------------------------------------------------------------------------
# find_similar — structure
# ---------------------------------------------------------------------------

class TestFindSimilarStructure:
    def test_returns_three_keys(self, index, first_row_query):
        results = find_similar(first_row_query, index, n_rows=N_ROWS)
        assert set(results.keys()) == {"cosine", "jaccard", "knn"}

    def test_each_entry_is_int_float_tuple(self, index, first_row_query):
        results = find_similar(first_row_query, index, n_rows=N_ROWS)
        for metric, hits in results.items():
            for row_idx, dist in hits:
                assert isinstance(row_idx, int),   f"{metric}: index must be int"
                assert isinstance(dist, float),    f"{metric}: distance must be float"
                assert dist >= 0.0,                f"{metric}: distance must be non-negative"

    def test_respects_top_k(self, index, first_row_query):
        results = find_similar(first_row_query, index, n_rows=N_ROWS, top_k=3)
        for metric, hits in results.items():
            assert len(hits) <= 3, f"{metric}: returned more than top_k=3 results"

    def test_results_sorted_closest_first(self, index, first_row_query):
        results = find_similar(first_row_query, index, n_rows=N_ROWS)
        for metric, hits in results.items():
            distances = [d for _, d in hits]
            assert distances == sorted(distances), f"{metric}: not sorted by distance"


# ---------------------------------------------------------------------------
# find_similar — correctness
# ---------------------------------------------------------------------------

class TestFindSimilarCorrectness:
    def test_exact_match_top_cosine(self, city_data, index):
        """A query identical to training row 0 should be its own nearest cosine neighbour."""
        query = {
            "cosine":  city_data.cosine_matrix[0:1].copy(),
            "jaccard": city_data.jaccard_matrix[0:1].copy(),
            "knn":     city_data.knn_matrix[0:1].copy(),
        }
        results = find_similar(query, index, n_rows=N_ROWS)
        top_idx, top_dist = results["cosine"][0]
        assert top_idx  == 0,    "Nearest cosine neighbour of itself should be index 0"
        assert top_dist < 1e-6,  "Cosine distance to itself should be ~0"

    def test_exact_match_top_knn(self, city_data, index):
        """Same check for euclidean distance."""
        query = {
            "cosine":  city_data.cosine_matrix[0:1].copy(),
            "jaccard": city_data.jaccard_matrix[0:1].copy(),
            "knn":     city_data.knn_matrix[0:1].copy(),
        }
        results = find_similar(query, index, n_rows=N_ROWS)
        top_idx, top_dist = results["knn"][0]
        assert top_idx  == 0,    "Nearest KNN neighbour of itself should be index 0"
        assert top_dist < 1e-6,  "Euclidean distance to itself should be ~0"

    def test_all_zero_jaccard_does_not_crash(self, city_data, index):
        """Degenerate all-zero Jaccard vector should return results, not raise."""
        query = {
            "cosine":  city_data.cosine_matrix[0:1].copy(),
            "jaccard": np.zeros((1, N_JACCARD)),
            "knn":     city_data.knn_matrix[0:1].copy(),
        }
        results = find_similar(query, index, n_rows=N_ROWS)
        assert len(results["jaccard"]) > 0

    def test_all_zero_jaccard_returns_max_distance(self, city_data, index):
        """All-zero vs any binary vector → Jaccard distance = 1.0 (no overlap)."""
        query = {
            "cosine":  city_data.cosine_matrix[0:1].copy(),
            "jaccard": np.zeros((1, N_JACCARD)),
            "knn":     city_data.knn_matrix[0:1].copy(),
        }
        results = find_similar(query, index, n_rows=N_ROWS)
        for _, dist in results["jaccard"]:
            assert dist == pytest.approx(1.0), "Jaccard(∅, any) should be 1.0"


# ---------------------------------------------------------------------------
# Hybrid scoring (_aggregate)
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_multi_metric_listing_ranks_first(self):
        """A listing in all 3 metrics at rank 0 should beat one in only 1 metric."""
        hits = {
            "cosine":  [(0, 0.1), (1, 0.5)],
            "jaccard": [(0, 0.2), (2, 0.6)],
            "knn":     [(0, 0.15), (3, 0.7)],
        }
        ranked = _aggregate(hits, top_k=2)
        assert ranked[0][0] == 0, "Listing 0 (in all 3 metrics) should rank first"

    def test_rank_based_scoring(self):
        """With top_k=3: rank-0 → 3 pts, rank-1 → 2 pts, rank-2 → 1 pt."""
        hits = {"cosine": [(10, 0.1), (20, 0.2), (30, 0.3)]}
        score_map = dict(_aggregate(hits, top_k=3))
        assert score_map[10] == 3
        assert score_map[20] == 2
        assert score_map[30] == 1

    def test_scores_accumulate_across_metrics(self):
        """Listing appearing rank-0 in two metrics should outscore rank-0 in one."""
        hits = {
            "cosine": [(99, 0.1)],
            "jaccard": [(99, 0.1)],
            "knn":     [(88, 0.1)],
        }
        score_map = dict(_aggregate(hits, top_k=5))
        assert score_map[99] == 10, "Rank-0 twice with top_k=5 → 5+5=10"
        assert score_map[88] == 5,  "Rank-0 once with top_k=5 → 5"

    def test_empty_hits_returns_empty_list(self):
        assert _aggregate({}, top_k=5) == []

    def test_output_sorted_descending(self):
        hits = {
            "cosine":  [(1, 0.1), (2, 0.5), (3, 0.9)],
            "jaccard": [(2, 0.2)],
        }
        ranked = _aggregate(hits, top_k=3)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_high_agreement_raises_confidence(self):
        """All comps appearing in all metrics → agreement component = 1.0."""
        hits = {
            "cosine":  [(0, 0.1), (1, 0.2)],
            "jaccard": [(0, 0.1), (1, 0.2)],
            "knn":     [(0, 0.1), (1, 0.2)],
        }
        agreement = _metric_agreement(hits, top_indices=[0, 1])
        assert agreement == pytest.approx(1.0)

    def test_no_agreement_lowers_confidence(self):
        """Each comp in only one metric → agreement = 1/3."""
        hits = {
            "cosine":  [(0, 0.1)],
            "jaccard": [(1, 0.1)],
            "knn":     [(2, 0.1)],
        }
        agreement = _metric_agreement(hits, top_indices=[0, 1, 2])
        assert agreement == pytest.approx(1 / 3)

    def test_tight_prices_high_consistency(self):
        prices = np.array([100.0, 101.0, 99.0, 100.5])
        assert _price_consistency(prices) > 0.95

    def test_wide_prices_low_consistency(self):
        prices = np.array([50.0, 300.0, 20.0, 400.0])
        assert _price_consistency(prices) < 0.5

    def test_confidence_in_unit_range(self):
        hits = {
            "cosine":  [(0, 0.1), (1, 0.2)],
            "jaccard": [(0, 0.3)],
            "knn":     [(1, 0.4), (2, 0.5)],
        }
        prices = np.array([120.0, 115.0, 110.0])
        conf = _compute_confidence(hits, [0, 1, 2], prices, n_requested=5)
        assert 0.0 <= conf <= 1.0

    def test_perfect_conditions_near_1(self):
        """Full agreement + near-identical prices + full coverage → confidence near 1."""
        hits = {
            "cosine":  [(i, 0.01 * i) for i in range(5)],
            "jaccard": [(i, 0.01 * i) for i in range(5)],
            "knn":     [(i, 0.01 * i) for i in range(5)],
        }
        prices = np.array([100.0, 100.5, 99.5, 100.2, 99.8])
        conf = _compute_confidence(hits, list(range(5)), prices, n_requested=5)
        assert conf > 0.85
