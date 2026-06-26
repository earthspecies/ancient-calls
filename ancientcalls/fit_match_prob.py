"""
The Pr(match | similarity) link function.

Maps a cosine similarity to the probability that two calls are the same call
type. Two forms, matching the paper:

- ``histogram``: non-parametric empirical match rate per similarity bin.
- ``logit``: logistic regression on the similarity.

A fitted link function serialises to a single portable JSON file (no pickle), so
the paper's fitted function can be shipped and dropped in via
``link_function_path``.
"""

from __future__ import annotations

import numpy as np

from ancientcalls import io

# Default similarity bins for the histogram link function (finer near 1.0, where
# matches concentrate). Matches the paper's binning.
DEFAULT_LEFT_EDGES = [0.0, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DEFAULT_RIGHT_EDGES = [0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]


class LinkFunction:
    """A fitted similarity -> match-probability function.

    Construct via :func:`fit_link_function` or :meth:`load`. Call
    :meth:`predict` with an array of similarities to get match probabilities.
    """

    def __init__(self, kind: str, params: dict) -> None:
        self.kind = kind
        self.params = params

    def predict(self, sims: np.ndarray) -> np.ndarray:
        sims = np.asarray(sims, dtype=float)
        if self.kind == "histogram":
            left = np.asarray(self.params["left_bin_edges"])
            right = np.asarray(self.params["right_bin_edges"])
            probs = np.asarray(self.params["probabilities"])
            # Bin index per similarity; clamp out-of-range values to the edge bins.
            idx = np.searchsorted(right, sims, side="right")
            idx = np.clip(idx, 0, len(probs) - 1)
            out = probs[idx]
            out = np.where(sims < left[0], probs[0], out)
            return out
        if self.kind == "logit":
            coef = float(self.params["coef"])
            intercept = float(self.params["intercept"])
            return 1.0 / (1.0 + np.exp(-(coef * sims + intercept)))
        raise ValueError(f"Unknown link function kind {self.kind!r}")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params}

    def save(self, path: io.PathLike) -> None:
        io.save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: io.PathLike) -> "LinkFunction":
        d = io.load_json(path)
        return cls(d["kind"], d["params"])


def _fit_histogram(sims: np.ndarray, matches: np.ndarray, left_edges: list[float], right_edges: list[float]) -> dict:
    """Empirical match rate per bin.

    Bins with no samples are filled by interpolating from neighbouring filled
    bins (the private code raised instead; interpolation is friendlier for
    arbitrary user data — a warning is printed).
    """
    probs: list[float] = []
    centers = [(lo + hi) / 2 for lo, hi in zip(left_edges, right_edges, strict=False)]
    filled_centers, filled_probs = [], []
    for lo, hi, c in zip(left_edges, right_edges, centers, strict=False):
        mask = (sims >= lo) & (sims < hi)
        if mask.sum() == 0:
            probs.append(np.nan)
        else:
            p = float(matches[mask].mean())
            probs.append(p)
            filled_centers.append(c)
            filled_probs.append(p)
    if not filled_centers:
        raise ValueError("No labelled pairs fell into any similarity bin.")
    arr = np.array(probs)
    if np.isnan(arr).any():
        print(f"Warning: {int(np.isnan(arr).sum())} empty histogram bin(s) filled by interpolation.")
        arr = np.interp(centers, filled_centers, filled_probs)
    return {"left_bin_edges": list(left_edges), "right_bin_edges": list(right_edges), "probabilities": arr.tolist()}


DEFAULT_LOGIT_C = 100.0  # inverse regularization strength; the paper used C=100.


def _fit_logit(sims: np.ndarray, matches: np.ndarray, C: float = DEFAULT_LOGIT_C) -> dict:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver="lbfgs", C=C)
    model.fit(sims.reshape(-1, 1), matches)
    return {"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def fit_link_function(
    similarities: np.ndarray,
    matches: np.ndarray,
    kind: str = "histogram",
    left_edges: list[float] | None = None,
    right_edges: list[float] | None = None,
    logit_C: float = DEFAULT_LOGIT_C,
) -> LinkFunction:
    """Fit a link function from labelled pairs.

    similarities: 1-D array of cosine similarities.
    matches: 1-D array of 0/1 match labels (same length).
    kind: "histogram" or "logit".
    logit_C: inverse regularization strength for the logit fit (the paper used
        100; ignored for the histogram).
    """
    sims = np.asarray(similarities, dtype=float)
    y = np.asarray(matches, dtype=int)
    if sims.shape != y.shape:
        raise ValueError(f"similarities {sims.shape} and matches {y.shape} must have the same shape")
    if kind == "histogram":
        params = _fit_histogram(sims, y, left_edges or DEFAULT_LEFT_EDGES, right_edges or DEFAULT_RIGHT_EDGES)
    elif kind == "logit":
        params = _fit_logit(sims, y, C=logit_C)
    else:
        raise ValueError(f"kind must be 'histogram' or 'logit', got {kind!r}")
    return LinkFunction(kind, params)


def fit_from_pairs_table(path: io.PathLike, kind: str = "histogram", logit_C: float = DEFAULT_LOGIT_C) -> LinkFunction:
    """Fit a link function from a labelled-pairs table with ``similarity`` and
    ``match`` columns (see docs/data-format.md). ``logit_C`` sets the logit's
    regularization (default 100, the paper's value)."""
    df = io.load_metadata(path)
    for col in ("similarity", "match"):
        if col not in df.columns:
            raise ValueError(f"labeled pairs table must have a {col!r} column; found {list(df.columns)}")
    return fit_link_function(df["similarity"].to_numpy(), df["match"].to_numpy(), kind=kind, logit_C=logit_C)
