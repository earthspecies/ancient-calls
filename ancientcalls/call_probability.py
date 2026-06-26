"""
Per-species call probabilities from similarities.

For each reference call and each species in the tree, take the maximum cosine
similarity between the reference call and that species' calls, then map it
through the fitted link function to get the probability the species has the call
type. A reference call's own species is fixed to probability 1.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from ancientcalls.fit_match_prob import LinkFunction


def call_keys_for(n_reference_calls: int) -> list[str]:
    """Short, plot-friendly keys for reference calls: ``call_1`` ... ``call_N``."""
    return [f"call_{i + 1}" for i in range(n_reference_calls)]


def compute_call_probabilities(
    sims: np.ndarray,
    species_per_example: np.ndarray,
    reference_species: np.ndarray,
    species_in_tree: list[str],
    link_function: LinkFunction,
    call_keys: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return ``{species: {call_key: probability}}``.

    sims: (n_reference, n_examples) cosine similarities.
    species_per_example: (n_examples,) species name (underscored) per dataset row.
    reference_species: (n_reference,) species (underscored) of each reference call.
    species_in_tree: species to produce probabilities for.
    """
    n_ref = sims.shape[0]
    if call_keys is None:
        call_keys = call_keys_for(n_ref)
    species_per_example = np.asarray(species_per_example)
    reference_species = np.asarray(reference_species)
    species_in_dataset = set(species_per_example.tolist())

    result: dict[str, dict[str, float]] = {}
    for s in tqdm(species_in_tree, desc="Call probabilities"):
        # Species in the tree but absent from the dataset get probability 0.
        if s not in species_in_dataset:
            result[s] = {ck: 0.0 for ck in call_keys}
            continue

        mask = species_per_example == s
        max_sims = sims[:, mask].max(axis=1)  # best similarity to species s, per reference call

        needs_pred = reference_species != s
        probs = np.zeros(n_ref)
        if needs_pred.any():
            probs[needs_pred] = link_function.predict(max_sims[needs_pred])
        probs[~needs_pred] = 1.0  # same-species reference calls are present by definition
        probs = np.clip(probs, 0.0, 1.0)
        result[s] = {ck: round(float(p), 4) for ck, p in zip(call_keys, probs, strict=False)}
    return result


def top_matched_clips(
    sims: np.ndarray,
    species_per_example: np.ndarray,
    example_ids: np.ndarray,
    species_in_tree: list[str],
    k: int,
    call_keys: list[str] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """For each reference call and species, the ``k`` most-similar clips.

    Returns ``{call_key: {species: [{example_id, similarity}, ...]}}`` — the
    evidence the verification tool shows so a human can judge each species match.
    """
    n_ref = sims.shape[0]
    if call_keys is None:
        call_keys = call_keys_for(n_ref)
    species_per_example = np.asarray(species_per_example)
    example_ids = np.asarray(example_ids)
    species_in_dataset = set(species_per_example.tolist())

    out: dict[str, dict[str, list[dict]]] = {ck: {} for ck in call_keys}
    for s in tqdm(species_in_tree, desc="Top matched clips"):
        if s not in species_in_dataset:
            continue
        col_idx = np.flatnonzero(species_per_example == s)
        sp_sims = sims[:, col_idx]  # (n_ref, n_species_clips)
        kk = min(k, sp_sims.shape[1])
        # Indices of the top-kk clips per reference call, best first.
        top = np.argsort(-sp_sims, axis=1)[:, :kk]
        for i, ck in enumerate(call_keys):
            clips = [
                {"example_id": str(example_ids[col_idx[j]]), "similarity": round(float(sp_sims[i, j]), 4)}
                for j in top[i]
            ]
            out[ck][s] = clips
    return out
