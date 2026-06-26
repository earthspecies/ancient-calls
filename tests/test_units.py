"""Focused per-module unit tests.

These lock in the behaviour verified by the parity tests. They are
deterministic and need no R, so they run anywhere the package imports.
"""

from __future__ import annotations

import numpy as np
import pytest
from ete3 import Tree

from commoncalls import call_probability, fit_match_prob, phylogeny, reconstruction, similarity
from commoncalls.config import PipelineConfig
from commoncalls.fit_match_prob import LinkFunction

# ── phylogeny ────────────────────────────────────────────────────────────────


def test_prune_to_species_keeps_subset_and_stays_ultrametric():
    # Non-ultrametric source: A,B at depth 3, C,D at depth 1.5.
    tree = Tree("((A:1,B:1):2,(C:1,D:1):0.5);", format=1)
    pruned = phylogeny.prune_to_species(tree, ["A", "B", "C"])
    assert set(pruned.get_leaf_names()) == {"A", "B", "C"}  # only requested + present
    depths = [pruned.get_distance(lf) for lf in pruned.get_leaves()]
    assert max(depths) - min(depths) < 1e-9  # re-scaled back to ultrametric


def test_prune_to_species_raises_when_fewer_than_two():
    tree = Tree("((A:1,B:1):2,(C:1,D:1):0.5);", format=1)
    with pytest.raises(ValueError, match="at least 2"):
        phylogeny.prune_to_species(tree, ["A"])  # only one requested species in tree
    with pytest.raises(ValueError, match="at least 2"):
        phylogeny.prune_to_species(tree, ["X", "Y"])  # none in tree


def test_build_species_tree_names_internal_nodes(tmp_path):
    src = tmp_path / "src.nwk"
    src.write_text("((A:1,B:1):2,(C:1,D:1):0.5);")
    out = tmp_path / "out.nwk"
    phylogeny.build_species_tree(src, out, ["A", "B", "C", "D"])
    tree = Tree(str(out), format=1)
    internal = [n for n in tree.traverse() if not n.is_leaf()]
    assert internal and all(n.name for n in internal)  # every internal node named


# ── fit_match_prob ───────────────────────────────────────────────────────────


def test_logit_predict_matches_closed_form():
    coef, intercept = 2.0, -1.0
    lf = LinkFunction("logit", {"coef": coef, "intercept": intercept})
    x = np.array([0.0, 0.3, 0.5, 0.9, 1.0])
    expected = 1.0 / (1.0 + np.exp(-(coef * x + intercept)))
    assert np.allclose(lf.predict(x), expected, atol=1e-12)


def test_histogram_bin_lookup_is_correct_and_monotone():
    lf = LinkFunction(
        "histogram", {"left_bin_edges": [0.0, 0.5], "right_bin_edges": [0.5, 1.01], "probabilities": [0.1, 0.9]}
    )
    # Hand values: left-inclusive / right-exclusive bins.
    assert lf.predict(np.array([0.0]))[0] == 0.1
    assert lf.predict(np.array([0.2]))[0] == 0.1
    assert lf.predict(np.array([0.5]))[0] == 0.9  # 0.5 falls in the second bin
    assert lf.predict(np.array([0.7]))[0] == 0.9
    grid = np.linspace(0, 1, 101)
    out = lf.predict(grid)
    assert np.all(np.diff(out) >= 0)  # non-decreasing


def test_histogram_empty_bin_interpolates_and_warns(capsys):
    # Few points -> several default bins empty -> interpolation + warning, no crash.
    sims = np.array([0.1, 0.45, 0.97])
    matches = np.array([0, 0, 1])
    lf = fit_match_prob.fit_link_function(sims, matches, kind="histogram")
    out = capsys.readouterr().out
    assert "empty histogram bin" in out
    assert not np.isnan(np.asarray(lf.params["probabilities"])).any()


# ── similarity ───────────────────────────────────────────────────────────────


def test_cosine_similarities_equals_matmul():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 8)).astype("float32")
    allm = rng.normal(size=(5, 8)).astype("float32")
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    allm /= np.linalg.norm(allm, axis=1, keepdims=True)
    assert np.allclose(similarity.cosine_similarities(ref, allm), ref @ allm.T, atol=1e-6)


def test_cosine_similarities_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimension mismatch|mismatch"):
        similarity.cosine_similarities(np.zeros((3, 4)), np.zeros((5, 6)))


# ── call_probability ─────────────────────────────────────────────────────────


def _callprob_fixture():
    sims = np.array(
        [
            [0.9, 0.8, 0.3, 0.2],  # ref1, species A
            [0.4, 0.5, 0.95, 0.1],
        ]
    )  # ref2, species B
    species_per_example = np.array(["A", "A", "B", "C"])
    reference_species = np.array(["A", "B"])
    species_in_tree = ["A", "B", "C", "D"]  # D absent from the dataset
    lf = LinkFunction(
        "histogram", {"left_bin_edges": [0.0, 0.5], "right_bin_edges": [0.5, 1.01], "probabilities": [0.1, 0.9]}
    )
    return sims, species_per_example, reference_species, species_in_tree, lf


def test_call_probabilities_special_cases():
    sims, spe, ref_sp, in_tree, lf = _callprob_fixture()
    res = call_probability.compute_call_probabilities(sims, spe, ref_sp, in_tree, lf)
    assert res["A"]["call_1"] == 1.0  # reference call's own species -> present
    assert res["B"]["call_2"] == 1.0
    assert res["D"] == {"call_1": 0.0, "call_2": 0.0}  # species absent from dataset -> 0
    for sp in res.values():
        for p in sp.values():
            assert 0.0 <= p <= 1.0  # clipped to [0,1]


def test_top_matched_clips_count_and_sorted():
    sims, spe, _, in_tree, _ = _callprob_fixture()
    example_ids = np.array(["A0", "A1", "B0", "C0"])
    clips = call_probability.top_matched_clips(sims, spe, example_ids, in_tree, k=2)
    a_clips = clips["call_1"]["A"]
    assert len(a_clips) == 2  # species A has 2 clips, k=2
    assert a_clips[0]["similarity"] >= a_clips[1]["similarity"]  # sorted desc
    assert len(clips["call_1"]["C"]) == 1  # only 1 clip available -> <= k


# ── reconstruction.node_ages ─────────────────────────────────────────────────


def test_node_ages_on_hand_tree():
    # Ultrametric: every leaf at depth 5. Internal depths: N1=3, N2=1, Root=0.
    tree = Tree("((A:2,B:2)N1:3,(C:4,D:4)N2:1)Root;", format=1)
    ages = reconstruction.node_ages(tree, ["A", "B", "C", "D"])
    assert abs(ages["Root"] - 5.0) < 1e-9  # age = max_leaf_depth - 0
    assert abs(ages["N1"] - 2.0) < 1e-9  # 5 - 3
    assert abs(ages["N2"] - 4.0) < 1e-9  # 5 - 1


def test_node_ages_degenerate_returns_empty():
    tree = Tree("((A:2,B:2)N1:3,(C:4,D:4)N2:1)Root;", format=1)
    assert reconstruction.node_ages(tree, ["A"]) == {}  # < 2 species


# ── config ───────────────────────────────────────────────────────────────────


def test_config_from_yaml_parses_modes(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "embeddings_path: e.npy\nmetadata_path: m.csv\ntree_path: t.nwk\n"
        "link_function_path: lf.json\n"
        "reconstruction_modes:\n"
        "  - mode: binary\n    leaf_threshold: 0.6\n    ancestor_threshold: 0.5\n"
        "  - mode: continuous\n    ancestor_threshold: 0.7\n"
    )
    cfg = PipelineConfig.from_yaml(cfg_path)
    assert [m.mode for m in cfg.reconstruction_modes] == ["binary", "continuous"]
    assert cfg.reconstruction_modes[0].leaf_threshold == 0.6
    assert cfg.reconstruction_modes[0].ancestor_threshold == 0.5
    assert cfg.reconstruction_modes[1].ancestor_threshold == 0.7


def test_config_validate_rejects_missing_link_function():
    cfg = PipelineConfig(embeddings_path="e", metadata_path="m", tree_path="t")
    with pytest.raises(ValueError, match="labeled_pairs_path|link_function_path"):
        cfg.validate()


def test_config_validate_rejects_bad_link_function():
    cfg = PipelineConfig(
        embeddings_path="e", metadata_path="m", tree_path="t", link_function_path="lf.json", link_function="bogus"
    )
    with pytest.raises(ValueError, match="link_function"):
        cfg.validate()


# ── verified reconstruction overrides ────────────────────────────────────────


def test_apply_verifications_overrides():
    from run_verified_reconstruction import apply_verifications

    call_probs = {"Sp_a": {"call_1": 0.3, "call_2": 0.4}, "Sp_b": {"call_1": 0.5, "call_2": 0.6}}
    export = {
        "verifier": "x",
        "verifications": {
            "call_1": {"Sp_a": {"status": "match"}, "Sp_b": {"status": "no_match"}},
            "call_2": {"Sp_a": {"status": "unsure"}, "Sp_z": {"status": "match"}},  # Sp_z not in run
        },
    }
    out, stats = apply_verifications(call_probs, export)
    assert out["Sp_a"]["call_1"] == 1.0  # match -> present
    assert out["Sp_b"]["call_1"] == 0.0  # no_match -> absent
    assert out["Sp_a"]["call_2"] == 0.4  # unsure -> unchanged
    assert stats == {"applied": 2, "skipped_unknown": 1, "left_unchanged": 1}
    assert call_probs["Sp_a"]["call_1"] == 0.3  # input not mutated
    # The inner map (without the wrapper) is accepted too.
    out2, _ = apply_verifications(call_probs, export["verifications"])
    assert out2["Sp_a"]["call_1"] == 1.0


def test_logit_C_default_is_100():
    from commoncalls.config import PipelineConfig
    from commoncalls.fit_match_prob import DEFAULT_LOGIT_C

    assert DEFAULT_LOGIT_C == 100.0
    assert PipelineConfig(embeddings_path="e", metadata_path="m", tree_path="t").logit_C == 100.0
