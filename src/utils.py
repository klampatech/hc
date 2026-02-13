import json
import numpy as np
from typing import Any, Dict


def safe_get(d: Dict, *keys, default=None) -> Any:
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, {})
        else:
            return default
    return d if d != {} else default


def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and batch of vectors.

    Args:
        query: shape (dim,)
        vectors: shape (n, dim), assumed pre-normalized

    Returns: similarities, shape (n,)
    """
    query_norm = query / np.linalg.norm(query)
    return vectors @ query_norm


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix of vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    return vectors / norms
