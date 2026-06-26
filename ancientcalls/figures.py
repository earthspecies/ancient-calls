"""
Summary figures: an age histogram across reference calls, and a phylogeny
heatmap of per-species call probabilities.

These are static matplotlib figures generated from the pipeline's own outputs
(``summary.json`` and ``call_probabilities.json``). The interactive,
per-call exploration lives in the local ``verify`` tool instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
from ete3 import Tree
from matplotlib.colors import LinearSegmentedColormap

from ancientcalls import io

# white (P=0) -> green (P=1), with grey for "no data".
CMAP = LinearSegmentedColormap.from_list("call_cmap", [(1.0, 1.0, 1.0), (0.25, 0.70, 0.25)])
NO_DATA_RGBA = (0.88, 0.88, 0.88, 1.0)


def mode_label(entry: dict) -> str:
    """Human-readable label for a reconstruction-mode summary entry."""
    if entry["mode"] == "binary":
        return f"binary (leaf {entry['leaf_threshold']}, ancestor {entry['ancestor_threshold']})"
    return f"continuous (ancestor {entry['ancestor_threshold']})"


def age_histogram(summary: dict[str, list[dict]], out_path: io.PathLike, bins: int = 20) -> Path:
    """Histogram of inferred ages, one panel per reconstruction mode."""
    by_mode: dict[str, list[float]] = {}
    for entries in summary.values():
        for e in entries:
            if e["age"] is not None:
                by_mode.setdefault(mode_label(e), []).append(e["age"])

    labels = sorted(by_mode)
    fig, axes = plt.subplots(1, max(1, len(labels)), figsize=(5 * max(1, len(labels)), 4), squeeze=False)
    for ax, label in zip(axes[0], labels, strict=False):
        ax.hist(by_mode[label], bins=bins, color="steelblue", edgecolor="white", rwidth=0.85)
        ax.set_xlabel("Inferred age (Myr)")
        ax.set_ylabel("Number of calls")
        ax.set_title(label, fontsize=10)
    fig.suptitle("Distribution of inferred call ages", fontsize=12)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _tree_layout(tree: Tree) -> tuple[dict, dict]:
    leaf_index = {lf.name: i for i, lf in enumerate(tree.get_leaves())}
    node_x, node_y = {}, {}
    for node in tree.traverse("preorder"):
        node_x[node] = 0 if node.is_root() else node_x[node.up] + node.dist
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            node_y[node] = leaf_index[node.name]
        else:
            ys = [node_y[c] for c in node.children if c in node_y]
            node_y[node] = float(np.mean(ys)) if ys else 0.0
    return node_x, node_y


def call_probability_heatmap(
    tree_path: io.PathLike,
    call_probs: dict[str, dict[str, float]],
    out_path: io.PathLike,
    call_order: list[str] | None = None,
    max_calls: int = 60,
) -> Path:
    """Phylogeny with one heatmap column per reference call (species × call probability)."""
    tree = io.load_tree(tree_path)
    leaves = tree.get_leaves()
    n = len(leaves)
    leaf_index = {lf.name: i for i, lf in enumerate(leaves)}

    if call_order is None:
        call_order = sorted({ck for v in call_probs.values() for ck in v})
    if len(call_order) > max_calls:
        print(f"Heatmap: showing first {max_calls} of {len(call_order)} calls (raise max_calls to show more).")
        call_order = call_order[:max_calls]

    node_x, node_y = _tree_layout(tree)
    fig, axes = plt.subplots(
        1,
        len(call_order) + 1,
        figsize=(7.5 + len(call_order) * 0.25, 16),
        gridspec_kw={"width_ratios": [8] + [0.5] * len(call_order)},
    )
    fig.patch.set_facecolor("white")

    segs = []
    for node in tree.traverse():
        x, y = node_x[node], node_y[node]
        if not node.is_root():
            segs.append([(node_x[node.up], y), (x, y)])
        if not node.is_leaf():
            child_ys = [node_y[c] for c in node.children]
            segs.append([(x, min(child_ys)), (x, max(child_ys))])
    axes[0].add_collection(mc.LineCollection(segs, colors="black", linewidths=0.6, alpha=0.7))
    max_x = max(node_x.values(), default=1)
    for lf in leaves:
        axes[0].text(max_x * 1.02, leaf_index[lf.name], lf.name, fontsize=7, va="center", ha="left")
    axes[0].set_xlim(-0.02 * max_x, max_x * 1.5)
    axes[0].set_ylim(n - 0.5, -0.5)
    axes[0].axis("off")
    axes[0].set_title("Phylogeny", fontsize=9)

    for idx, key in enumerate(call_order, start=1):
        ax = axes[idx]
        img = np.ones((n, 1, 4))
        img[:, 0, :] = NO_DATA_RGBA
        for sp, calls in call_probs.items():
            if key in calls and sp in leaf_index:
                img[leaf_index[sp], 0, :] = CMAP(calls[key])
        ax.imshow(img, aspect="auto", origin="upper", extent=[0, 1, n - 0.5, -0.5], interpolation="nearest")
        ax.set_title(key, fontsize=5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Call probability", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
