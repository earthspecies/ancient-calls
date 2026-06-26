"""Robustness / edge-case tests: #1 fit-from-pairs,
#3 species mismatch, #6 top_k_clips=0.

These drive the full pipeline, so they skip when Rscript is unavailable.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from make_example_fixture import make_example_dataset

from commoncalls import io, reconstruction
from commoncalls.config import PipelineConfig
from commoncalls.fit_match_prob import LinkFunction
from commoncalls.pipeline import run

REPO = Path(__file__).resolve().parents[1]

needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not available")


def _base_config(tmp_path, **overrides):
    p = make_example_dataset(tmp_path, calls_per_species=15)
    kw = dict(
        embeddings_path=str(p["embeddings"]),
        metadata_path=str(p["metadata"]),
        tree_path=str(p["tree"]),
        link_function_path=str(p["link_function"]),  # pre-fit fn (overridden by #1)
        work_dir=str(tmp_path / "work"),
        n_reference_calls=5,
    )
    kw.update(overrides)
    return PipelineConfig(**kw), p


# ── #1: fit the link function from labelled pairs (no pre-fit function) ────────


@needs_r
@pytest.mark.parametrize("kind", ["histogram", "logit"])
def test_fit_from_labeled_pairs(tmp_path, kind):
    # A labelled-pairs table: similarity + binary match (both classes present).
    sims = np.linspace(0.05, 1.0, 60)
    matches = (sims > 0.75).astype(int)
    pairs_path = tmp_path / "pairs.csv"
    pd.DataFrame({"similarity": sims, "match": matches}).to_csv(pairs_path, index=False)

    config, _ = _base_config(
        tmp_path,
        link_function_path=None,  # drop the pre-fit fn; fit from pairs instead
        labeled_pairs_path=str(pairs_path),
        link_function=kind,
    )
    run(config)

    saved = config.work / "link_function.json"
    assert saved.exists()  # pipeline fits and saves the link function
    assert LinkFunction.load(saved).kind == kind


# ── #3: tree species and dataset species differ — subset to the intersection ──


@needs_r
def test_species_mismatch_subsets_to_intersection(tmp_path):
    config, p = _base_config(tmp_path)
    # Tree has Sp_a/b/c (in dataset) + Sp_x (not in dataset); dataset also has
    # Sp_d/e/f (not in tree). Intersection is {Sp_a, Sp_b, Sp_c}.
    Path(p["tree"]).write_text("((Sp_a:1,Sp_b:1):2,(Sp_c:1.5,Sp_x:1.5):1);")

    summary = run(config)
    call_probs = io.load_json(config.call_probabilities_path)
    assert set(call_probs) == {"Sp_a", "Sp_b", "Sp_c"}  # Sp_x and Sp_d/e/f excluded
    assert summary  # ran without crashing


# ── #6: top_k_clips = 0 — no matched_clips.json; verify builder still runs ─────


@needs_r
def test_top_k_clips_zero(tmp_path):
    config, _ = _base_config(tmp_path, top_k_clips=0)
    summary = run(config)
    assert not config.matched_clips_path.exists()  # no clips file written
    assert summary

    # The verify builder must still produce data, with empty clip lists.
    spec = importlib.util.spec_from_file_location("bvd", REPO / "verify" / "build_verify_data.py")
    bvd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bvd)
    out_dir = Path(config.work_dir) / "verify_data"
    bvd.build(config, out_dir, audio_root=None, copy_audio=False)

    assert (out_dir / "data.json").exists()
    call_files = list((out_dir / "calls").glob("*.json"))
    assert call_files
    detail = io.load_json(call_files[0])
    assert all(row["clips"] == [] for row in detail["species"])  # clips empty, not crashing


# ── verify UI export -> re-run reconstruction on verified data ────────────────


@needs_r
def test_verified_reconstruction_flow(tmp_path):
    from run_verified_reconstruction import apply_verifications

    config, _ = _base_config(tmp_path)
    run(config)
    call_probs = io.load_json(config.call_probabilities_path)
    species = list(call_probs)

    # Verify call_1 as present in two species; re-run reconstruction on the result.
    export = {"verifications": {"call_1": {species[0]: {"status": "match"}, species[1]: {"status": "match"}}}}
    new_cp, stats = apply_verifications(call_probs, export)
    assert stats["applied"] == 2
    assert new_cp[species[0]]["call_1"] == 1.0

    summary = reconstruction.run(
        config.pruned_tree_path, new_cp, config.reconstruction_modes, tmp_path / "verif", n_workers=1
    )
    assert summary  # verified reconstruction runs end-to-end
