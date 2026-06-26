"""End-to-end smoke test on a tiny synthetic, contract-conforming dataset.

Generates embeddings, metadata, a dated tree, and a link function in a temp dir,
runs the full pipeline, and checks that ages come out. Requires R with the ape,
castor, and jsonlite packages (the reconstruction backend); the reconstruction
assertions are skipped if Rscript is unavailable.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from make_example_fixture import NEWICK, SPECIES, make_example_dataset  # shared synthetic generator

from ancientcalls import fit_match_prob
from ancientcalls.config import PipelineConfig
from ancientcalls.pipeline import run

CALLS_PER_SPECIES = 5


def _make_dataset(tmp_path):
    p = make_example_dataset(tmp_path, calls_per_species=CALLS_PER_SPECIES)
    return p["embeddings"], p["metadata"], p["tree"], p["link_function"]


def test_pipeline_end_to_end(tmp_path):
    emb_path, meta_path, tree_path, lf_path = _make_dataset(tmp_path)
    config = PipelineConfig(
        embeddings_path=str(emb_path),
        metadata_path=str(meta_path),
        tree_path=str(tree_path),
        link_function_path=str(lf_path),
        work_dir=str(tmp_path / "work"),
        n_reference_calls=8,
    )

    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available; skipping reconstruction assertions")

    summary = run(config)

    # Every reference call should appear with one entry per reconstruction mode.
    assert len(summary) == 8
    for entries in summary.values():
        modes = {(e["mode"], e["ancestor_threshold"]) for e in entries}
        assert ("binary", 0.6) in modes
        assert ("continuous", 0.6) in modes

    # The pipeline writes its artifacts.
    assert (tmp_path / "work" / "species_tree.nwk").exists()
    assert (tmp_path / "work" / "call_probabilities.json").exists()

    # At least one call should trace back to a non-trivial ancestral age (the
    # shared call type across Sp_a/b/c).
    ages = [e["age"] for entries in summary.values() for e in entries if e["age"] is not None]
    assert ages, "expected at least one non-null inferred age"
    assert max(ages) > 0


def test_reconstruction_worker_count_invariance(tmp_path):
    """Parallelism must be a pure speed knob: results identical across n_workers."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    from ancientcalls import phylogeny, reconstruction
    from ancientcalls.config import ReconstructionMode

    (tmp_path / "src.nwk").write_text(NEWICK)
    tree_path = tmp_path / "species_tree.nwk"
    phylogeny.build_species_tree(tmp_path / "src.nwk", tree_path, SPECIES)

    abc = {"Sp_a", "Sp_b", "Sp_c"}
    call_probs = {sp: {"call_1": 0.9 if sp in abc else 0.1, "call_2": 0.8 if sp in abc else 0.2} for sp in SPECIES}
    modes = [ReconstructionMode("continuous", 0.6), ReconstructionMode("binary", 0.6, 0.6)]

    s1 = reconstruction.run(tree_path, call_probs, modes, tmp_path / "r1", n_workers=1)
    s4 = reconstruction.run(tree_path, call_probs, modes, tmp_path / "r4", n_workers=4)
    assert s1 == s4


def test_link_function_roundtrip(tmp_path):
    sims = np.array([0.1, 0.2, 0.85, 0.9, 0.95, 0.99])
    matches = np.array([0, 0, 1, 1, 1, 1])
    lf = fit_match_prob.fit_link_function(sims, matches, kind="histogram")
    p = tmp_path / "lf.json"
    lf.save(p)
    lf2 = fit_match_prob.LinkFunction.load(p)
    pred = lf2.predict(np.array([0.05, 0.92]))
    assert pred.shape == (2,)
    assert pred[1] >= pred[0]  # higher similarity -> higher (or equal) match prob


def _cfg(tmp_path, **kw):
    return PipelineConfig(embeddings_path="e", metadata_path="m", tree_path="t", link_function_path="l", **kw)


def test_pinned_reference_ids_used_verbatim(tmp_path):
    from ancientcalls.pipeline import _select_reference_ids

    ids = [f"x{i}" for i in range(10)]
    pinned = ["x3", "x7", "x1"]  # arbitrary subset + order
    (tmp_path / "ref.json").write_text(json.dumps(pinned))
    cfg = _cfg(tmp_path, reference_call_ids_path=str(tmp_path / "ref.json"))
    # Exact and order-preserving (call_key assignment depends on order).
    assert _select_reference_ids(ids, cfg) == pinned


def test_pinned_reference_ids_missing_id_raises(tmp_path):
    from ancientcalls.pipeline import _select_reference_ids

    (tmp_path / "bad.json").write_text(json.dumps(["x3", "not_in_dataset"]))
    cfg = _cfg(tmp_path, reference_call_ids_path=str(tmp_path / "bad.json"))
    with pytest.raises(ValueError, match="not in the dataset"):
        _select_reference_ids([f"x{i}" for i in range(10)], cfg)


def test_sampling_is_deterministic_without_pin(tmp_path):
    from ancientcalls.pipeline import _select_reference_ids

    ids = [f"x{i}" for i in range(10)]
    cfg = _cfg(tmp_path, n_reference_calls=3, random_seed=42)
    assert _select_reference_ids(ids, cfg) == _select_reference_ids(ids, cfg)
