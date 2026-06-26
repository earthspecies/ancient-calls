# Example data

This folder holds the inputs `configs/example.yaml` points at. The files are
**generated, not committed** (synthetic data, and `*.npy` is gitignored).
Generate them, then run the pipeline end-to-end:

```bash
python scripts/make_example_fixture.py        # writes the four files below
ancientcalls-run --config configs/example.yaml # runs the full pipeline
```

- `embeddings.npy` — `(N, D)` float32, L2-normalized per row
- `metadata.csv` — row-aligned, with `example_id` and `species` columns
- `tree.nwk` — dated Newick whose leaves are the species
- `match_function.json` — a pre-fit `Pr(match|similarity)` link function
  (or provide labelled pairs and fit one)

See [`../../docs/data-format.md`](../../docs/data-format.md) for the full
contract and [`../../scripts/make_example_fixture.py`](../../scripts/make_example_fixture.py)
for a concrete, readable example of how to produce it. The same generator backs
the smoke test (`tests/test_smoke.py`), so the test runs without anything here.

To bring your own data, drop your real files here (or point the `*_path` keys in
the config elsewhere) and skip the generator step.
