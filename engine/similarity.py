import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config.settings import SIMILARITY_TOP_K
from data.preprocessor import CityData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index container
# ---------------------------------------------------------------------------

@dataclass
class SimilarityIndex:
    cosine_nn: NearestNeighbors    # brute cosine on full feature vector
    jaccard_nn: NearestNeighbors   # brute jaccard on binary vector (no neighbourhood OHE)
    knn_nn: NearestNeighbors       # euclidean on scaled numerics only


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(city_data: CityData) -> SimilarityIndex:
    """Fit three NearestNeighbors models on the city's pre-computed matrices.

    Called once per city at startup.  Fitting is fast (brute-force just stores
    the data; no tree to build) so these are not cached to disk.
    """
    n = len(city_data.df)
    k = min(SIMILARITY_TOP_K, n)
    logger.info("Building similarity index for %d listings (top_k=%d) ...", n, k)

    cosine_nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
    ).fit(city_data.cosine_matrix)

    # Jaccard requires binary input — cast float 0/1 to bool.
    jaccard_nn = NearestNeighbors(
        n_neighbors=k,
        metric="jaccard",
        algorithm="brute",
    ).fit(city_data.jaccard_matrix.astype(bool))

    knn_nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",   # lets sklearn pick ball_tree / kd_tree / brute by size
    ).fit(city_data.knn_matrix)

    logger.info("Similarity index ready.")
    return SimilarityIndex(cosine_nn=cosine_nn, jaccard_nn=jaccard_nn, knn_nn=knn_nn)


def find_similar(
    query_vectors: Dict[str, np.ndarray],
    index: SimilarityIndex,
    n_rows: int,
    top_k: int = SIMILARITY_TOP_K,
) -> Dict[str, List[Tuple[int, float]]]:
    """Return the top-K nearest listings for each of the three metrics.

    Parameters
    ----------
    query_vectors : output of preprocessor.preprocess_query()
        {"cosine": (1, d_c), "jaccard": (1, d_j), "knn": (1, d_k)}
    index : fitted SimilarityIndex for the target city
    n_rows : number of listings in the city (to clamp k safely)
    top_k : how many neighbours to return per metric

    Returns
    -------
    {
        "cosine":  [(row_idx, distance), ...],  # sorted closest-first
        "jaccard": [(row_idx, distance), ...],
        "knn":     [(row_idx, distance), ...],
    }
    Each list has at most top_k entries.
    """
    k = min(top_k, n_rows)

    # --- Cosine ---
    dists, idxs = index.cosine_nn.kneighbors(query_vectors["cosine"], n_neighbors=k)
    cosine_hits: List[Tuple[int, float]] = list(
        zip(idxs[0].tolist(), dists[0].tolist())
    )

    # --- Jaccard — query vector must be boolean to match the fitted index ---
    jac_query = query_vectors["jaccard"].astype(bool)
    dists, idxs = index.jaccard_nn.kneighbors(jac_query, n_neighbors=k)
    jaccard_hits: List[Tuple[int, float]] = list(
        zip(idxs[0].tolist(), dists[0].tolist())
    )

    # --- KNN (euclidean on scaled numerics) ---
    dists, idxs = index.knn_nn.kneighbors(query_vectors["knn"], n_neighbors=k)
    knn_hits: List[Tuple[int, float]] = list(
        zip(idxs[0].tolist(), dists[0].tolist())
    )

    return {
        "cosine":  cosine_hits,
        "jaccard": jaccard_hits,
        "knn":     knn_hits,
    }
