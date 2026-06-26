"""
End-to-end pipeline orchestration.

A single in-process function runs every step, driven by one ``PipelineConfig`` —
replacing the private repo's chain of subprocesses wired together with ~15
environment variables. Run from the CLI with ``ancientcalls-run --config cfg.yaml``.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from ancientcalls import call_probability, io, phylogeny, reconstruction, similarity
from ancientcalls.config import PipelineConfig
from ancientcalls.fit_match_prob import LinkFunction, fit_from_pairs_table


def _select_reference_ids(ids: list[str], config: PipelineConfig) -> list[str]:
    """Choose the reference-call id list.

    If ``reference_call_ids_path`` is set, load that exact list (order
    preserved) and verify every id is present in the dataset; this pins the
    reference set for reproducibility and parity. Otherwise sample
    ``n_reference_calls`` deterministically from ``random_seed`` (or use all).
    """
    if config.reference_call_ids_path is not None:
        pinned = io.load_json(config.reference_call_ids_path)
        id_set = set(ids)
        missing = [r for r in pinned if r not in id_set]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(pinned)} pinned reference id(s) are not in the dataset "
                f"(after subsetting to tree species); e.g. {missing[:3]}"
            )
        return list(pinned)
    rng = random.Random(config.random_seed)
    if config.n_reference_calls is not None and config.n_reference_calls < len(ids):
        return rng.sample(ids, k=config.n_reference_calls)
    return list(ids)


def run(config: PipelineConfig) -> dict[str, list[dict]]:
    """Run the full pipeline and return the per-call age summary.

    Writes the pruned tree, reference-call ids, call probabilities, and
    reconstruction results under ``config.work_dir``.
    """
    config.validate()
    config.work.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs.
    print("Loading embeddings + metadata...")
    embeddings = io.load_npy(config.embeddings_path)
    metadata = io.load_metadata(config.metadata_path)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"metadata rows ({len(metadata)}) != embedding rows ({embeddings.shape[0]})")
    species = metadata[config.species_column].str.replace(" ", "_")

    # 2. Build the species chronogram (prune to the dataset species unless a
    #    species list is supplied).
    if config.species_list_path is not None:
        species_list = [s.replace(" ", "_") for s in io.load_json(config.species_list_path)]
    else:
        species_list = sorted(set(species))
    phylogeny.build_species_tree(config.tree_path, config.pruned_tree_path, species_list)
    species_in_tree = io.species_from_tree(config.pruned_tree_path)
    tree_species = set(species_in_tree)

    # 3. Subset to examples whose species made it into the tree.
    in_tree = species.isin(tree_species).to_numpy()
    embeddings = embeddings[in_tree]
    metadata = metadata.loc[in_tree].reset_index(drop=True)
    species = species[in_tree].reset_index(drop=True)
    print(f"  {len(metadata)} calls across {len(tree_species)} tree species")

    # 4. Choose reference calls (pinned list if configured, else sample).
    ids = metadata[config.id_column].tolist()
    reference_ids = _select_reference_ids(ids, config)
    io.save_json(reference_ids, config.reference_calls_path)
    pinned = " (pinned)" if config.reference_call_ids_path is not None else ""
    print(f"  {len(reference_ids)} reference calls{pinned}")

    id_to_idx = {eid: i for i, eid in enumerate(ids)}
    reference_idxs = [id_to_idx[r] for r in reference_ids]
    reference_species = species.to_numpy()[reference_idxs]

    # 5. Cosine similarities (reference calls × all calls).
    print("Computing cosine similarities...")
    sims = similarity.cosine_similarities(embeddings[reference_idxs], embeddings)

    # 6. Link function: load a pre-fit one, or fit from labelled pairs.
    if config.link_function_path is not None:
        link_fn = LinkFunction.load(config.link_function_path)
        print(f"  loaded {link_fn.kind} link function")
    else:
        link_fn = fit_from_pairs_table(config.labeled_pairs_path, kind=config.link_function, logit_C=config.logit_C)
        link_fn.save(config.work / "link_function.json")
        print(f"  fit {link_fn.kind} link function from {config.labeled_pairs_path}")

    # 7. Per-species call probabilities.
    call_keys = call_probability.call_keys_for(len(reference_ids))
    call_probs = call_probability.compute_call_probabilities(
        sims, species.to_numpy(), reference_species, species_in_tree, link_fn, call_keys
    )
    io.save_json(call_probs, config.call_probabilities_path)
    io.save_json(dict(zip(call_keys, reference_ids, strict=False)), config.work / "call_key_to_example_id.json")

    # Top matched clips per (call, species) — evidence for the verification tool.
    if config.top_k_clips > 0:
        clips = call_probability.top_matched_clips(
            sims,
            species.to_numpy(),
            metadata[config.id_column].to_numpy(),
            species_in_tree,
            config.top_k_clips,
            call_keys,
        )
        io.save_json(clips, config.matched_clips_path)

    # 8. Ancestral-state reconstruction + age inference.
    summary = reconstruction.run(
        config.pruned_tree_path, call_probs, config.reconstruction_modes, config.results_dir, n_workers=config.n_workers
    )

    _report(summary)
    return summary


def _report(summary: dict[str, list[dict]]) -> None:
    ages = [e["age"] for entries in summary.values() for e in entries if e["age"] is not None]
    print(f"\nDone. {len(summary)} calls analysed; {len(ages)} (call × mode) results with a non-null age.")
    if ages:
        print(f"  age range: {min(ages):.2f} – {max(ages):.2f} Myr (median {float(np.median(ages)):.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ancient-calls phylogenetic age pipeline.")
    parser.add_argument("--config", required=True, help="Path to a pipeline YAML config.")
    args = parser.parse_args()
    run(PipelineConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
