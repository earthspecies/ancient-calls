"""
Ancestral-state reconstruction and call-age inference.

The reconstruction itself runs in R (``reconstruction.R``) via ``Rscript`` — much
easier to install than rpy2. This module builds the per-call traits, drives R
once per analysis, and computes node ages in Python from the reconstructed
likelihoods.

Age of an internal node = max_leaf_depth − depth(node) on the tree pruned to the
species carrying the trait (time before present, in the tree's branch-length
units). A call's reported age is the oldest internal node whose reconstructed
presence likelihood ≥ the ancestor threshold.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from ete3 import Tree
from tqdm import tqdm

from ancientcalls import io
from ancientcalls.config import ReconstructionMode

HERE = Path(__file__).resolve().parent
R_SCRIPT = HERE / "reconstruction.R"


# ── trait construction ──────────────────────────────────────────────────────


def make_trait(call_probs: dict[str, dict[str, float]], call_key: str) -> dict[str, float]:
    """Raw per-species probability for one call (continuous trait)."""
    return {sp: v[call_key] for sp, v in call_probs.items() if call_key in v}


def threshold_trait(trait: dict[str, float], threshold: float) -> dict[str, float]:
    """Convert a continuous trait to binary presence/absence at ``threshold``."""
    return {sp: (1.0 if v >= threshold else 0.0) for sp, v in trait.items()}


# ── age computation ─────────────────────────────────────────────────────────


def node_ages(full_tree: Tree, species: list[str]) -> dict[str, float]:
    """Map internal node name -> age, on the tree pruned to ``species``.

    Age = max_leaf_depth − depth(node). Returns {} if the pruned tree is
    degenerate (fewer than 2 species or zero depth).
    """
    if len(species) < 2:
        return {}
    pruned = full_tree.copy()
    pruned.prune(list(species), preserve_branch_length=True)
    for node in pruned.traverse():
        if node.dist == 0 and not node.is_root():
            node.dist = 1.0
    max_depth = max(pruned.get_distance(lf) for lf in pruned.get_leaves())
    if max_depth == 0:
        return {}
    internal = set(pruned.traverse()) - set(pruned.get_leaves())
    return {n.name: max_depth - pruned.get_distance(n) for n in internal}


def age_of_oldest_present(node_vals: dict[str, float], name_to_age: dict[str, float], threshold: float) -> float | None:
    """Oldest node age among nodes with likelihood ≥ threshold, or None."""
    ages = [name_to_age[n] for n, v in node_vals.items() if v >= threshold and n in name_to_age]
    return max(ages) if ages else None


# ── R driver ────────────────────────────────────────────────────────────────


def _run_r(tree_path: io.PathLike, traits: dict[str, dict[str, float]], mode: str) -> dict[str, dict[str, float]]:
    """Run reconstruction.R for a batch of calls; return {call_key: {node: lik}}.

    ``mode`` is "binary" or "continuous". Traits are written to a temp JSON; the
    tree must be a local path (R reads it directly).
    """
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found. Install R and the packages: ape, castor, jsonlite "
            '(in R: install.packages(c("ape", "castor", "jsonlite"))).'
        )
    with tempfile.TemporaryDirectory() as tmp:
        traits_path = Path(tmp) / "traits.json"
        out_path = Path(tmp) / "out.json"
        io.save_json(traits, traits_path)
        proc = subprocess.run(
            ["Rscript", str(R_SCRIPT), str(tree_path), str(traits_path), mode, str(out_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"reconstruction.R failed:\n{proc.stdout}\n{proc.stderr}")
        return io.load_json(out_path)


def _run_r_parallel(
    tree_path: io.PathLike, traits: dict[str, dict[str, float]], mode: str, n_workers: int | None
) -> dict[str, dict[str, float]]:
    """Run reconstruction.R across several processes, one per chunk of calls.

    Each call is independent, so results are identical to a single sequential
    run regardless of how calls are chunked. Parallelism is at the process level
    (each R process is single-threaded), so we cap workers at the CPU count.
    """
    keys = list(traits)
    # Default leaves one core free (matches the private rpy2 worker pool).
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    workers = n_workers if n_workers is not None else default_workers
    workers = max(1, min(workers, len(keys)))
    if workers == 1:
        return _run_r(tree_path, traits, mode)

    chunk_size = math.ceil(len(keys) / workers)
    chunks = [{k: traits[k] for k in keys[i : i + chunk_size]} for i in range(0, len(keys), chunk_size)]
    out: dict[str, dict[str, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        # subprocess.run releases the GIL while R runs, so threads give true
        # process-level parallelism without pickling the inputs.
        for res in ex.map(lambda c: _run_r(tree_path, c, mode), chunks):
            out.update(res)
    return out


# ── orchestration ───────────────────────────────────────────────────────────


def run(
    tree_path: io.PathLike,
    call_probs: dict[str, dict[str, float]],
    modes: list[ReconstructionMode],
    results_dir: io.PathLike,
    n_workers: int | None = None,
) -> dict[str, list[dict]]:
    """Run all reconstruction analyses and return a per-call age summary.

    For each call key, the summary lists one entry per mode:
    ``{"mode", "leaf_threshold", "ancestor_threshold", "age"}``. Raw per-node
    likelihoods are also written to ``results_dir/<analysis_name>/raw-results.json``.

    Binary modes that share a leaf threshold reuse one R run; differing ancestor
    thresholds are then applied cheaply to the cached node likelihoods.

    n_workers: parallel R processes (default: CPU count − 1, leaving one core
    free). Calls are independent, so the result is identical regardless of
    worker count.
    """
    full_tree = io.load_tree(tree_path)
    tree_species = {lf.name for lf in full_tree.get_leaves()}
    call_keys = sorted({ck for probs in call_probs.values() for ck in probs.keys()})
    print(f"Reconstructing {len(call_keys)} calls over {len(tree_species)} species")

    results_dir = Path(results_dir)
    summary: dict[str, list[dict]] = defaultdict(list)

    # Group analyses so R runs once per (mode kind, leaf threshold).
    continuous = [m for m in modes if m.mode == "continuous"]
    binary_by_leaf: dict[float, list[ReconstructionMode]] = defaultdict(list)
    for m in modes:
        if m.mode == "binary":
            binary_by_leaf[m.leaf_threshold].append(m)

    age_cache: dict[frozenset, dict[str, float]] = {}

    def ages_for(species: list[str]) -> dict[str, float]:
        key = frozenset(species)
        if key not in age_cache:
            age_cache[key] = node_ages(full_tree, species)
        return age_cache[key]

    def analyse(kind: str, leaf_threshold: float | None, ancestor_thresholds: list[float]) -> None:
        # Build per-call traits restricted to tree species.
        traits: dict[str, dict[str, float]] = {}
        for ck in call_keys:
            t = {sp: v for sp, v in make_trait(call_probs, ck).items() if sp in tree_species}
            if kind == "binary":
                if sum(1 for v in t.values() if v >= leaf_threshold) < 2:
                    continue  # too few present species to reconstruct
                t = threshold_trait(t, leaf_threshold)
            else:
                if not t or sum(t.values()) == 0:
                    continue
            traits[ck] = t
        if not traits:
            return

        node_vals_by_call = _run_r_parallel(tree_path, traits, kind, n_workers)

        for at in ancestor_thresholds:
            name = f"{kind}_leaf{leaf_threshold}_ancestor{at}" if kind == "binary" else f"continuous_ancestor{at}"
            raw: dict[str, dict] = {}
            for ck in tqdm(traits, desc=name):
                node_vals = node_vals_by_call.get(ck, {})
                name_to_age = ages_for(list(traits[ck].keys()))
                age = age_of_oldest_present(node_vals, name_to_age, at)
                raw[ck] = {"trait": traits[ck], "node_vals": node_vals, "age": age}
                summary[ck].append(
                    {"mode": kind, "leaf_threshold": leaf_threshold, "ancestor_threshold": at, "age": age}
                )
            io.save_json(raw, results_dir / name / "raw-results.json")

    for m in continuous:
        analyse("continuous", None, [m.ancestor_threshold])
    for leaf_threshold, group in binary_by_leaf.items():
        analyse("binary", leaf_threshold, [m.ancestor_threshold for m in group])

    io.save_json(summary, results_dir / "summary.json")
    return dict(summary)
