#!/usr/bin/env python
"""Generate a tiny synthetic, contract-conforming example dataset.

This is the single source of truth for the toy dataset: it backs both the
end-to-end smoke test (``tests/test_smoke.py``) and the runnable example wired
up by ``configs/example.yaml``. Nothing here is real data -- it exists so you
can run the whole pipeline (and read a concrete example of the input contract
in ``docs/data-format.md``) before bringing your own embeddings.

    python scripts/make_example_fixture.py            # -> data/example/*
    ancientcalls-run --config configs/example.yaml     # run end-to-end

The "science" is rigged so the result is interpretable: Sp_a/Sp_b/Sp_c share a
strong common "call type" direction (so it reconstructs to a deep common
ancestor), while Sp_d/Sp_e/Sp_f get their own unrelated directions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ancientcalls import fit_match_prob, io

REPO_ROOT = Path(__file__).resolve().parent.parent

SPECIES = ["Sp_a", "Sp_b", "Sp_c", "Sp_d", "Sp_e", "Sp_f"]
# Dated tree; internal nodes are left unnamed (the pipeline names them) and it is
# deliberately not perfectly ultrametric -- build_species_tree re-scales leaves.
NEWICK = "((Sp_a:1,Sp_b:1):2,((Sp_c:1.5,Sp_d:1.5):1,(Sp_e:2,Sp_f:2):0.5):0.5);"
SHARED_SPECIES = ("Sp_a", "Sp_b", "Sp_c")


def make_example_dataset(
    out_dir: Path | str,
    *,
    calls_per_species: int = 20,
    dim: int = 8,
    seed: int = 0,
) -> dict[str, Path]:
    """Write embeddings/metadata/tree/link-function into ``out_dir``.

    Returns a dict of the four written paths (keys: embeddings, metadata, tree,
    link_function).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shared = rng.normal(size=dim)  # the common ancestral "call type" direction
    per_species_base = {s: rng.normal(size=dim) for s in SPECIES}

    rows, embs = [], []
    for s in SPECIES:
        for j in range(calls_per_species):
            v = 0.3 * per_species_base[s] + 0.1 * rng.normal(size=dim)
            if s in SHARED_SPECIES:
                v = v + 1.2 * shared  # strong shared component
            embs.append(v)
            rows.append({"example_id": f"{s}_{j}", "species": s})
    emb = np.asarray(embs, dtype="float32")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2-normalize per row

    paths = {
        "embeddings": out_dir / "embeddings.npy",
        "metadata": out_dir / "metadata.csv",
        "tree": out_dir / "tree.nwk",
        "link_function": out_dir / "match_function.json",
    }
    io.save_npy(emb, paths["embeddings"])
    pd.DataFrame(rows).to_csv(paths["metadata"], index=False)
    paths["tree"].write_text(NEWICK)
    # Monotonic logit link: high similarity -> high match probability.
    fit_match_prob.LinkFunction("logit", {"coef": 12.0, "intercept": -6.0}).save(paths["link_function"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic example dataset.")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "example"),
        help="Output directory (default: data/example).",
    )
    parser.add_argument("--calls-per-species", type=int, default=20)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = make_example_dataset(args.out, calls_per_species=args.calls_per_species, dim=args.dim, seed=args.seed)
    n = len(SPECIES) * args.calls_per_species
    print(f"Wrote {n} examples across {len(SPECIES)} species (dim {args.dim}) to {args.out}")
    for key, p in paths.items():
        print(f"  {key:13s} {p}")


if __name__ == "__main__":
    main()
