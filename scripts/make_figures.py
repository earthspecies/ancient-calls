#!/usr/bin/env python
"""Generate summary figures from a finished pipeline run.

    python scripts/make_figures.py --config configs/example.yaml

Reads the run's summary.json and call_probabilities.json under work_dir and
writes figures to work_dir/figures/.
"""

from __future__ import annotations

import argparse

from commoncalls import figures, io
from commoncalls.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Make summary figures for a pipeline run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-calls", type=int, default=60, help="Max calls to show in the heatmap.")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    fig_dir = config.work / "figures"

    summary = io.load_json(config.results_dir / "summary.json")
    out = figures.age_histogram(summary, fig_dir / "age_histogram.png")
    print(f"Wrote {out}")

    call_probs = io.load_json(config.call_probabilities_path)
    out = figures.call_probability_heatmap(
        config.pruned_tree_path, call_probs, fig_dir / "call_probability_heatmap.png", max_calls=args.max_calls
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
