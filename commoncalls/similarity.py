"""
Cosine similarity between reference calls and the full dataset.

Embeddings are L2-normalized (see the data contract), so cosine similarity is a
plain dot product. Uses PyTorch on GPU if available, else NumPy — both optional
to the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np


def cosine_similarities(reference_embeddings: np.ndarray, all_embeddings: np.ndarray) -> np.ndarray:
    """Return the (n_reference, n_all) matrix of cosine similarities.

    Both inputs must be L2-normalized row-wise. The result is
    ``reference_embeddings @ all_embeddings.T``.
    """
    if reference_embeddings.shape[1] != all_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: reference {reference_embeddings.shape} vs all {all_embeddings.shape}"
        )
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
        if device is not None:
            ref = torch.from_numpy(reference_embeddings).to(device)
            allm = torch.from_numpy(all_embeddings).to(device)
            return torch.mm(ref, allm.T).cpu().numpy()
    except ImportError:
        pass
    return reference_embeddings @ all_embeddings.T
