#!/usr/bin/env python
"""Re-run reconstruction after applying human verifications from the verify UI.

The verify tool (`verify/verify.html`) exports a JSON of per-(call, species)
judgements. This script overrides the call probabilities with those judgements
(``match`` -> 1.0, ``no_match`` -> 0.0; ``unsure`` and unjudged left as-is) and
re-runs ancestral-state reconstruction, producing a "verified" age summary
alongside the original run.

    python scripts/run_verified_reconstruction.py \\
        --config configs/example.yaml \\
        --verifications verifications_alice_2026-06-24.json

Writes to ``<work_dir>/reconstruction_verified/`` by default:
  - call_probabilities_verified.json   the overridden probabilities
  - summary.json + per-mode raw-results (from reconstruction.run)

Plotting the verified ages is left to you (see scripts/make_figures.py for the
base run). Multi-verifier conflict resolution is out of scope: pass one export.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from commoncalls import io, reconstruction
from commoncalls.config import PipelineConfig

_OVERRIDE = {"match": 1.0, "no_match": 0.0}  # "unsure"/unjudged -> leave unchanged


def apply_verifications(
    call_probs: dict[str, dict[str, float]],
    verifications: dict,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Return a copy of ``call_probs`` with verified judgements applied.

    ``verifications`` is the verify-UI export (the full object with a
    ``"verifications"`` key, or just the inner ``{call: {species: {...}}}`` map).
    Returns ``(new_call_probs, stats)`` where stats counts applied/skipped/left.
    """
    inner = verifications.get("verifications", verifications) if isinstance(verifications, dict) else {}
    out = copy.deepcopy(call_probs)
    stats = {"applied": 0, "skipped_unknown": 0, "left_unchanged": 0}
    for call_key, species_map in inner.items():
        for species, info in species_map.items():
            new = _OVERRIDE.get((info or {}).get("status"))
            if new is None:
                stats["left_unchanged"] += 1  # unsure / no decision
                continue
            if species in out and call_key in out[species]:
                out[species][call_key] = new
                stats["applied"] += 1
            else:
                stats["skipped_unknown"] += 1  # (call, species) not in this run
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run reconstruction with verified overrides.")
    parser.add_argument("--config", required=True, help="The pipeline config of the original run.")
    parser.add_argument("--verifications", required=True, help="Exported verifications JSON from verify.html.")
    parser.add_argument("--out", default=None, help="Output dir (default: <work_dir>/reconstruction_verified).")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    call_probs = io.load_json(config.call_probabilities_path)
    verifications = io.load_json(args.verifications)

    new_call_probs, stats = apply_verifications(call_probs, verifications)
    print(
        f"Verifications applied: {stats['applied']}  "
        f"(left unchanged: {stats['left_unchanged']}, skipped unknown: {stats['skipped_unknown']})"
    )

    out_dir = Path(args.out) if args.out else config.work / "reconstruction_verified"
    out_dir.mkdir(parents=True, exist_ok=True)
    io.save_json(new_call_probs, out_dir / "call_probabilities_verified.json")

    summary = reconstruction.run(
        config.pruned_tree_path, new_call_probs, config.reconstruction_modes, out_dir, n_workers=config.n_workers
    )
    ages = [e["age"] for entries in summary.values() for e in entries if e["age"] is not None]
    print(f"Verified reconstruction: {len(summary)} calls; {len(ages)} non-null ages. Wrote {out_dir}")


if __name__ == "__main__":
    main()
