"""
I/O helpers for the pipeline.

All file access goes through these helpers so the pipeline works identically on
the local filesystem (the default) or on remote object stores such as Google
Cloud Storage. Remote access is provided by ``fsspec``; ``gs://`` paths require
the optional ``gcs`` extra (``pip install commoncalls[gcs]``).

There is no hardcoded bucket or server anywhere — a path is local unless it
carries an explicit protocol (e.g. ``gs://``, ``s3://``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
from Bio import Phylo
from ete3 import Tree

PathLike = str | os.PathLike


def open_file(path: PathLike, mode: str = "r"):
    """Open a local or remote file.

    Uses ``fsspec`` so ``gs://``/``s3://`` paths work transparently when the
    relevant extra is installed; plain paths resolve to the local filesystem.
    For local writes, parent directories are created automatically.
    """
    path = str(path)
    if "://" not in path and ("w" in mode or "a" in mode or "x" in mode):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return fsspec.open(path, mode)


def load_json(path: PathLike) -> Any:
    with open_file(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    with open_file(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_npy(path: PathLike) -> np.ndarray:
    """Load a ``.npy`` array (local or remote)."""
    with open_file(path, "rb") as f:
        return np.load(f)


def save_npy(array: np.ndarray, path: PathLike) -> None:
    with open_file(path, "wb") as f:
        np.save(f, array)


def load_metadata(path: PathLike) -> pd.DataFrame:
    """Load the per-example metadata table.

    Supports ``.csv``, ``.parquet``, and ``.json`` (a list of records). The
    returned frame is row-aligned with the embeddings array. See
    ``docs/data-format.md`` for the required/optional columns.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        with open_file(path, "r") as f:
            return pd.read_csv(f)
    if suffix == ".parquet":
        with open_file(path, "rb") as f:
            return pd.read_parquet(f)
    if suffix == ".json":
        return pd.DataFrame(load_json(path))
    raise ValueError(f"Unsupported metadata format {suffix!r} for {path!r} (use .csv, .parquet, or .json)")


def load_tree(path: PathLike) -> Tree:
    """Load a Newick tree with ete3 (format=1, internal node names preserved).

    Zero-length non-root branches are bumped to 1.0 so they remain visible in
    plots and traversals, matching the original pipeline behaviour.
    """
    with open_file(path, "r") as f:
        newick = f.read()
    tree = Tree(newick, format=1)
    for node in tree.traverse():
        if node.dist == 0 and not node.is_root():
            node.dist = 1.0
    return tree


def save_tree(tree: Tree, path: PathLike) -> None:
    """Write a Newick tree (internal node names included)."""
    path = str(path)
    if "://" not in path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tree.write(outfile=path, format=1)
        return
    # Remote write: serialise then stream out.
    with open_file(path, "w") as f:
        f.write(tree.write(format=1))


def species_from_tree(path: PathLike) -> list[str]:
    """Return the leaf (species) names of a Newick tree."""
    with open_file(path, "r") as f:
        tree = Phylo.read(f, "newick")
    return [t.name for t in tree.get_terminals()]
