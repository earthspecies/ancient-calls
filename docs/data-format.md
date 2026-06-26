# Input data format (the contract)

The pipeline is data-agnostic: it does not depend on Xeno-Canto, BirdNET, or any
particular embedding model. You bring four things. Anything that conforms to the
shapes below will run.

All paths in the config may be local (default) or a remote URL. I/O goes through
`fsspec`, which dispatches on the URL scheme — install the matching extra
(`gs://` → `gcs`, `s3://` → `s3`, `az://` → `azure`, or `remote` for all) and
authenticate with that provider's standard mechanism
(`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_PROFILE`, `AZURE_STORAGE_*`). Local and
remote paths can be mixed freely in one config.

---

## 1. Embeddings — `embeddings_path`

A NumPy array saved as `.npy`:

- shape `(N, D)`: one row per call/example, `D` = embedding dimension
- dtype `float32`
- **L2-normalized per row** (unit norm), so cosine similarity is a dot product

```python
import numpy as np
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
np.save("embeddings.npy", emb.astype("float32"))
```

How you produce embeddings is up to you (any audio model, or a future built-in
producer — see the `embedding` extra). The pipeline only ever sees this array.

## 2. Metadata — `metadata_path`

A table **row-aligned** to the embeddings array (row `i` describes embedding row
`i`). Accepted formats: `.csv`, `.parquet`, or `.json` (list of records).

| column        | required | description                                  |
|---------------|----------|----------------------------------------------|
| `example_id`  | yes      | unique string id for the call                |
| `species`     | yes      | `Genus_species` (underscores), matches tree  |
| `order`       | no       | taxonomic order                              |
| `family`      | no       | taxonomic family                             |
| `genus`       | no       | taxonomic genus                              |
| `audio_fp`    | no       | path/URL to source audio (for the verify UI) |

Column names for the id and species fields are configurable (`id_column`,
`species_column`).

## 3. Match-probability link function

The pipeline converts a cosine similarity into a probability that two calls are
the same call type, `Pr(match | similarity)`. Provide **one** of:

- `labeled_pairs_path` — a table of labelled pairs to fit the function from:

  | column       | description                              |
  |--------------|------------------------------------------|
  | `similarity` | cosine similarity of the pair `[-1, 1]`  |
  | `match`      | `1` if same call type, else `0`          |

  Choose the form with `link_function: histogram` (binned empirical rate) or
  `logit` (regularized logistic regression). For `logit`, the inverse
  regularization strength is `logit_C` (default **100**, the value used in the
  paper); set it in the config to change. `logit_C` is ignored for `histogram`.

- `link_function_path` — a pre-fit link function to drop in and skip fitting.

### Where do labelled pairs come from?

You produce them on your side — there is no annotation tool in this repo (that is
out of scope). A labelled pair is **two calls, the human
judgement of whether they are the same call type (`match` 0/1), and their cosine
similarity**. In the paper, an annotator reviewed candidate matched pairs and
labelled each; the similarities came from the same embeddings used in the run.

> **The link function is embedding-specific.** It models the similarity
> *distribution of your embedding model*, so it must be fit on similarities from
> **the same embeddings you run the pipeline on**. Do not reuse a function fit on
> different embeddings (e.g. another model, or the paper's BirdNET embeddings) —
> the mapping would be wrong. This is why no pre-fit function is shipped: bring
> your own labels (or a function you fit on your own embeddings).

## 4. Dated tree — `tree_path`

An ultrametric, time-calibrated phylogeny in Newick format whose **leaf names
match the `species` column** (underscores, e.g. `Columbina_talpacoti`). Branch
lengths are interpreted as time; ages are reported in those units (millions of
years for a TimeTree.org chronogram).

The paper's tree is the Aves chronogram exported from
[TimeTree.org](https://timetree.org), with tip labels mapped to GBIF canonical
species names to match the metadata. Export the tree for your taxon there (or use
any time-calibrated Newick) and rename tips to your `species` values if needed.

- If you also set `species_list_path` (a JSON list of species), the tree is
  pruned to those species and re-scaled before analysis.
- Otherwise the tree is used as given. Internal nodes are named automatically if
  unnamed (`NamedNode_i`) so reconstructed likelihoods can be tracked per node.

A practical source is a TimeTree.org export for your clade. Note that TimeTree
uses NCBI names; if your `species` use a different taxonomy (e.g. GBIF), convert
the leaf names to match before running.

---

## Minimal config

```yaml
embeddings_path: data/example/embeddings.npy
metadata_path:   data/example/metadata.csv
tree_path:       data/example/tree.nwk
link_function_path: data/example/match_function.json   # or labeled_pairs_path

work_dir: work
n_reference_calls: 100

reconstruction_modes:
  - mode: binary
    leaf_threshold: 0.6
    ancestor_threshold: 0.6
  - mode: continuous
    ancestor_threshold: 0.6
```

### Remote inputs (any fsspec scheme)

Point any path at a remote URL; outputs can be remote too. Example reading inputs
from S3 (`pip install commoncalls[s3]`, credentials via `AWS_PROFILE` etc.):

```yaml
embeddings_path: s3://my-bucket/calls/embeddings.npy
metadata_path:   s3://my-bucket/calls/metadata.parquet
tree_path:       gs://other-bucket/trees/timetree.nwk   # mixing schemes is fine
link_function_path: data/example/match_function.json    # and local
work_dir: s3://my-bucket/runs/run1
```

## What is NOT in scope

Upstream steps from our study — audio segmentation, sound-event detection, and
the species filters (≥5 recording sources, restriction to the GBIF Aves tree) —
ran before this pipeline and are not reproduced here. Start from already-filtered
embeddings + metadata + species list.
