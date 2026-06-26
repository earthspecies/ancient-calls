"""
Build the species chronogram used for reconstruction.

Given a dated tree (e.g. a TimeTree.org export) and the set of species in the
dataset, prune the tree to those species, re-scale pruned leaves so the tree
stays ultrametric, and name every internal node so reconstructed likelihoods can
be tracked per node downstream.
"""

from __future__ import annotations

from Bio import Phylo
from ete3 import Tree

from ancientcalls import io


def prune_to_species(tree: Tree, species: list[str]) -> Tree:
    """Prune a dated tree to ``species`` and keep it ultrametric.

    Only species present in both the tree and the list are kept. After pruning
    (preserving branch lengths), each leaf's terminal branch is extended so all
    leaves again sit at the maximum original depth — preserving the chronology.
    """
    keep = set(tree.get_leaf_names()) & set(species)
    if len(keep) < 2:
        raise ValueError(f"Only {len(keep)} of {len(species)} species are in the tree; need at least 2.")

    original_depths = {lf.name: tree.get_distance(lf) for lf in tree.get_leaves() if lf.name in keep}
    target_depth = max(original_depths.values())

    tree.prune(list(keep), preserve_branch_length=True)
    for leaf in tree.get_leaves():
        leaf.dist += target_depth - tree.get_distance(leaf)
    return tree


def name_internal_nodes(tree_path: io.PathLike) -> None:
    """Give every unnamed internal node a stable name, in place on disk.

    Uses Bio.Phylo to match the original pipeline's Newick formatting. Internal
    node confidence values are cleared so names serialise as strings rather than
    floats.
    """
    tree = Phylo.read(str(tree_path), "newick")
    for i, node in enumerate(tree.get_nonterminals()):
        if not node.name:
            node.name = f"NamedNode_{i}"
        node.confidence = None
    Phylo.write(tree, str(tree_path), "newick")


def build_species_tree(
    source_tree_path: io.PathLike,
    out_path: io.PathLike,
    species: list[str] | None = None,
) -> None:
    """Produce the analysis chronogram and write it to ``out_path``.

    If ``species`` is given, the source tree is pruned to those species and
    re-scaled; otherwise it is copied as-is. Either way, internal nodes are named
    so per-node reconstruction likelihoods can be tracked.

    Species names are normalised to use underscores (``Genus species`` ->
    ``Genus_species``) to match leaf labels.
    """
    tree = io.load_tree(source_tree_path)
    if species is not None:
        species = [s.replace(" ", "_") for s in species]
        tree = prune_to_species(tree, species)
    print(f"Species tree: {len(tree.get_leaf_names())} leaves")
    io.save_tree(tree, out_path)
    name_internal_nodes(out_path)
