# ancient-calls

Estimate the **evolutionary age of shared animal calls** from audio embeddings
and a dated phylogeny. This is the analysis pipeline accompanying the paper
*(citation TBD)*.

Given a set of embedded calls and a time-calibrated species tree, the pipeline:

1. samples *reference calls* (or uses a pinned list, for reproducible reruns),
2. computes their cosine similarity to every call in the dataset,
3. converts similarity into a per-species probability that the call type is
   present, via a fitted `Pr(match | similarity)` link function,
4. runs ancestral-state reconstruction (R: `ape::ace` / `castor::asr_mk_model`)
   over the tree, and
5. reports, for each reference call, the age of the oldest ancestral node where
   the call is inferred present — plus figures and a local verification UI.

It is **data-agnostic**: bring your own embeddings, metadata, and dated tree.
Nothing is tied to a specific dataset or embedding model.

## What you bring

Four inputs, all local files or remote URLs. The full contract — with column
names, dtypes, and worked examples — is in
[`docs/data-format.md`](docs/data-format.md); the essentials:

| input | what | format |
|------|------|--------|
| **embeddings** | one row per call | `(N, D)` float32 `.npy`, **L2-normalized per row** |
| **metadata** | one row per call, row-aligned to the embeddings | `.csv` / `.parquet` / `.json` with `example_id` + `species` (`Genus_species`); optional `order`, `family`, `genus`, `audio_fp` |
| **link function** | turns similarity into `Pr(match)` | *either* a table of labelled pairs (`similarity`, `match`) to fit from, *or* a pre-fit link function to load |
| **dated tree** | time-calibrated species tree | ultrametric Newick whose leaf names match the `species` column |

**Dated tree.** The paper uses the Aves chronogram from
[TimeTree.org](https://timetree.org) — a public database of divergence times,
freely exportable for a taxon. Tip labels were mapped to GBIF canonical species
names so they match the `species` column (underscored, e.g. `Columbina_talpacoti`).
Any TimeTree.org export — or any ultrametric, time-calibrated Newick whose leaves
match your species — works; the tree itself is not redistributed here.

Upstream steps from the study (audio segmentation, sound-event detection, the
species filters) are **not** part of this pipeline — start from already-filtered
embeddings + metadata. See [`docs/embedding.md`](docs/embedding.md) for producing
embeddings.

## Install

```bash
git clone https://github.com/earthspecies/ancient-calls.git && cd ancient-calls
pip install -e .            # core pipeline
pip install -e ".[gcs]"     # + read/write gs:// paths
pip install -e ".[s3]"      # + s3://      (or [azure] for az://, [remote] for all)
pip install -e ".[dev]"     # + tests/lint
```

Requires Python 3.11. The reconstruction step calls **R** via `Rscript` — install
R and three packages:

```r
install.packages(c("ape", "castor", "jsonlite"))
```

Any input/output path may be a remote URL — I/O goes through `fsspec`, which
dispatches on the URL scheme. Install the matching extra (`gs://` → `gcs`,
`s3://` → `s3`, `az://` → `azure`) and authenticate with that provider's
standard mechanism (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE` /
`AWS_ACCESS_KEY_ID`, `AZURE_STORAGE_*`). No code changes — just point a config
path at the URL.

## Try it with no data

Generate a tiny synthetic, contract-conforming dataset and run the whole thing
end-to-end — no data of your own required. This is the fastest way to see the
inputs, the outputs, and the verify UI before wiring up your own files.

```bash
python scripts/make_example_fixture.py           # writes data/example/*
commoncalls-run --config configs/example.yaml     # full run (needs R)
python scripts/make_figures.py --config configs/example.yaml
python verify/build_verify_data.py --config configs/example.yaml
python -m http.server --directory work/verify_data 8000   # → http://localhost:8000
```

The fixture is rigged so the result is interpretable (three species share a
common call type that reconstructs to a deep ancestor). The same fixture backs
the test suite:

```bash
pytest                      # needs the [dev] extra; end-to-end tests skip without R
```

## Run on your own data

Point a config at your four inputs — copy `configs/example.yaml` and edit the
paths. A minimal config:

```yaml
embeddings_path: path/to/embeddings.npy
metadata_path:   path/to/metadata.csv
tree_path:       path/to/tree.nwk

# Provide ONE of these:
link_function_path: path/to/match_function.json   # a pre-fit link function
# labeled_pairs_path: path/to/labeled_pairs.csv    # ...or fit from labelled pairs
# link_function: histogram                          #    (logit | histogram)

work_dir: work
n_reference_calls: 100
random_seed: 42
# reference_call_ids_path: path/to/ids.json        # pin reference calls for reproducible reruns

reconstruction_modes:
  - mode: binary
    leaf_threshold: 0.6
    ancestor_threshold: 0.6
  - mode: continuous
    ancestor_threshold: 0.6
```

Then run the same three steps as above (`commoncalls-run` → `make_figures.py` →
`build_verify_data.py`). All outputs land under the config's `work_dir/`:

```
work/
  species_tree.nwk            pruned, dated chronogram (named internal nodes)
  call_probabilities.json     per-species call probability, per reference call
  reconstruction/             raw node likelihoods + summary.json (ages per mode)
  figures/                    age_histogram.png, call_probability_heatmap.png
  verify_data/                self-contained verify.html + index.html + data
```

See [`docs/methods.md`](docs/methods.md) for what each step computes.

## Explore the results

The verify tool runs **fully locally** — no server, no cloud. Open it after
`build_verify_data.py`:

- **`index.html`** — overview table of every reference call and its inferred age
  per reconstruction mode; sort and switch modes.
- **`verify.html`** — per-call detail: the reference call on the phylogeny (tips
  coloured by call probability, internal nodes by reconstructed likelihood), the
  species ranked by probability with their matched audio clips, and the reported
  age per mode. Mark each species match / no-match / unsure and **Export** a
  verification JSON.

To feed those human judgements back in, re-run reconstruction with the verified
probabilities (`match` → present, `no_match` → absent, `unsure` left as-is):

```bash
python scripts/run_verified_reconstruction.py \
    --config configs/example.yaml --verifications verifications_*.json
```

This writes a parallel `reconstruction_verified/` (overridden probabilities +
ages); plotting the verified ages is left to you.

## Layout

| path | what |
|------|------|
| `commoncalls/` | the installable pipeline package |
| `scripts/` | CLI helpers (run, figures, verified reconstruction, example fixture) |
| `verify/` | self-contained local HTML tools (no server needed) |
| `docs/` | input-data contract + method notes |
| `data/example/` | `make_example_fixture.py` writes the toy dataset here |

## License

MIT — see [`LICENSE`](LICENSE).

---

> **A note on provenance.** This repository was packaged from an internal,
> exploratory research repo — both for clarity (a clean, runnable extraction of
> the analysis pipeline) and for compliance with the underlying data licenses.
> The refactor was carried out largely with the help of
> [Claude Code](https://claude.com/claude-code). The pipeline has been tested
> end-to-end, but given the extent of the refactor it may still contain bugs;
> please report anything you find via the issue tracker.
