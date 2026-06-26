"""
Pipeline configuration.

A single ``PipelineConfig`` dataclass replaces the private repo's split
dataset/server YAML files and the ~15 environment variables that were passed
between subprocess steps. Load one from a YAML file with
``PipelineConfig.from_yaml(path)``.

All paths are local by default. Any path may instead be a remote URL (e.g.
``gs://...``) if the optional ``gcs`` extra is installed — the I/O layer handles
both transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ReconstructionMode:
    """One ancestral-state reconstruction analysis.

    mode: "binary" (ape::ace on thresholded presence/absence) or "continuous"
          (castor::asr_mk_model on raw probabilities used as tip priors).
    leaf_threshold: probability cutoff turning a species present/absent
          (binary mode only; ignored for continuous).
    ancestor_threshold: an internal node counts as "present" when its
          reconstructed likelihood >= this value; the call's age is the oldest
          such node.
    """

    mode: str
    ancestor_threshold: float = 0.6
    leaf_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("binary", "continuous"):
            raise ValueError(f"mode must be 'binary' or 'continuous', got {self.mode!r}")
        if self.mode == "binary" and self.leaf_threshold is None:
            raise ValueError("binary reconstruction mode requires a leaf_threshold")


@dataclass
class PipelineConfig:
    """Everything the pipeline needs for one run.

    Inputs (the data contract — see docs/data-format.md):
        embeddings_path:   (N, D) float32 .npy, L2-normalized per row.
        metadata_path:     row-aligned table with example_id + species columns.
        tree_path:         dated ultrametric Newick whose leaves are species.
                           If species_list_path is also given, this tree is
                           pruned to those species; otherwise it is used as-is.

    Match-probability link function (one of):
        labeled_pairs_path: pairs of (similarity, match label) to fit from.
        link_function_path: a pre-fit link function to load instead of fitting.
        link_function:      "histogram" or "logit" (used when fitting).

    Run settings:
        work_dir:          where all intermediate + output artifacts are written.
        n_reference_calls: number of reference calls to sample (None = all).
        random_seed:       seed for reference-call sampling.
        reference_call_ids_path: a JSON list of example_ids to use as the
                           reference-call set, in order, instead of sampling.
                           When set, n_reference_calls and random_seed are
                           ignored. Use for exact reproducibility (re-run a prior
                           run) and for parity testing against another pipeline.
        reconstruction_modes: analyses to run.
    """

    # Inputs
    embeddings_path: str
    metadata_path: str
    tree_path: str
    species_list_path: str | None = None

    # Match-probability link function
    labeled_pairs_path: str | None = None
    link_function_path: str | None = None
    link_function: str = "histogram"
    # Inverse regularization strength when fitting a logit link function (the
    # paper used 100). Ignored for the histogram and when loading a pre-fit fn.
    logit_C: float = 100.0

    # Run settings
    work_dir: str = "work"
    n_reference_calls: int | None = None
    random_seed: int = 42
    # Pin the reference-call set instead of sampling (reproducibility / parity).
    reference_call_ids_path: str | None = None
    species_column: str = "species"
    id_column: str = "example_id"
    # Per (reference call, species), keep this many top-similarity clips for the
    # verification tool. Set to 0 to skip (smaller outputs, no verify support).
    top_k_clips: int = 5
    # Parallel R processes for reconstruction (None = CPU count − 1, leaving one
    # core free). Each process is single-threaded and loads the tree, so mind
    # memory for very large trees. Results are independent of this value.
    n_workers: int | None = None
    reconstruction_modes: list[ReconstructionMode] = field(
        default_factory=lambda: [
            ReconstructionMode(mode="binary", leaf_threshold=0.6, ancestor_threshold=0.6),
            ReconstructionMode(mode="continuous", ancestor_threshold=0.6),
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        modes = raw.pop("reconstruction_modes", None)
        cfg = cls(**raw)
        if modes is not None:
            cfg.reconstruction_modes = [ReconstructionMode(**m) for m in modes]
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.link_function not in ("histogram", "logit"):
            raise ValueError(f"link_function must be 'histogram' or 'logit', got {self.link_function!r}")
        if self.labeled_pairs_path is None and self.link_function_path is None:
            raise ValueError(
                "Provide either labeled_pairs_path (to fit Pr(match|sim)) or "
                "link_function_path (a pre-fit link function)."
            )

    # Derived output locations, all under work_dir.
    @property
    def work(self) -> Path:
        return Path(self.work_dir)

    @property
    def pruned_tree_path(self) -> Path:
        return self.work / "species_tree.nwk"

    @property
    def reference_calls_path(self) -> Path:
        return self.work / "reference_call_ids.json"

    @property
    def call_probabilities_path(self) -> Path:
        return self.work / "call_probabilities.json"

    @property
    def matched_clips_path(self) -> Path:
        return self.work / "matched_clips.json"

    @property
    def results_dir(self) -> Path:
        return self.work / "reconstruction"
