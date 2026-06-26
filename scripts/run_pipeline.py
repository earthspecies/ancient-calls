#!/usr/bin/env python
"""Run the ancient-calls phylogenetic age pipeline from a YAML config.

Usage: python scripts/run_pipeline.py --config configs/example.yaml
"""

from ancientcalls.pipeline import main

if __name__ == "__main__":
    main()
